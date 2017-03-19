import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
print(cudnn.benchmark)
cudnn.benchmark = True
from torch.autograd import Variable
from torchvision import models, transforms
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import cPickle
from glob import glob
import sys
import os

import huva
from huva import plate_localizer, clip, rfcn, LazyImage, make_multiple
from huva.np_util import *
from huva.th_util import th_get_jet, get_model_param_norm, MonitoredAdam

from data import db2
from data import traffic_mixins

if __name__=='__main__':
    db = db2.Database.load('data/db2.pkl')

batch_size = 64
num_epoch = 50
input_H = 208
input_W = 208
downsample_times = 4
label_H = input_H / downsample_times
label_W = input_W / downsample_times


black_white = True   # whether we are running a black&white model instead of RGB
num_input_cn = 1 if black_white else 3
model_name = None    # name of the model, used for archiving and evaluationg
model = None         # actual model
criterion = None
optimizer = None

file_folder = os.path.dirname(os.path.realpath(__file__))

"""
TODO:
* Use black-white images
- Bring in trucks
- Filter out small crops for training
- Expand bottom of auto-label crop
- Add CLAHE
- Add stronger augmentation
  * image contrast enhancement
  * color channel swap
  * Horizontal flip
  - black-white
  - color_space conversion
  - brightness adjustment
  - Minor rotations
  - Stretch (crop with unequal sides, then resize to input_H by input_W)
- Monitor param norm and update norm
- Add weight decay and lr annealing

"""


"""
*************************************** Dataset ************************************************
"""


class CropsDataset():
    def __init__(self, path_crops_info, unpack=False):
        self.crops_info = cPickle.load(open(path_crops_info, 'rb'))
        if unpack:
            self.crops_info = [pic_info for frame_info in self.crops_info for pic_info in frame_info]
        random.shuffle(self.crops_info)
    def __len__(self):
        return len(self.crops_info)
    def __getitem__(self, i):
        """
        1. Load image
        2. Take a random crop
        3. Perform other augmentation
        4. Generate label
        """
        global black_white
        fname, bboxes = self.crops_info[i]
        # load the image in BGR order, and keep it this way
        if black_white:
            img_np = cv2.imread(fname, 0)
            H, W = img_np.shape[:2]
            img_np = img_np.reshape(H,W,1) # (H,W) -> (H,W,1)
        else:
            img_np = cv2.imread(fname)
            H, W = img_np.shape[:2]
        """ taking a random crop from the image """
        if True: # crop it
            paste_y = 0
            """ the case where all we have is a narrow, frontal view of the car """
            if W/float(H) > 1.6:
                crop_len_W = int(W * 0.75)
                crop_len_H = H
                crop_x = random.randint(0, W - crop_len_W)
                crop_y = 0
                paste_y = random.randint(0, crop_len_W - crop_len_H)
                """ create a square image, and paste the region we want into it """
                img_crop = np.zeros((crop_len_W, crop_len_W, num_input_cn), np.uint8)
                img_crop[paste_y:paste_y+crop_len_H, :crop_len_W, :] = \
                        img_np[crop_y:crop_y+crop_len_H, crop_x:crop_x+crop_len_W, :]
            else:
                min_side = min(H,W)
                max_side = max(H,W)
                if max_side/float(min_side) > 1.5:
                    crop_len = min_side
                else:
                    crop_len = int(min_side * random.uniform(0.6, 0.8))
                crop_x = random.randint(0, W - crop_len)
                crop_y = random.randint(0, H - crop_len)
                img_crop = img_np[crop_y:crop_y+crop_len, crop_x:crop_x+crop_len, :]
        """ other augmentation """
        if not black_white and random.randint(0,1):  # swap channels
            img_crop = swap_channels(img_crop)
        """
        if black_white and random.randint(0,1): # invert image
            img_crop = 255 - img_crop
        """
        if random.randint(0,1):  # normalize
            img_crop = normalize_image(img_crop)
        flip = random.randint(0,1)
        if flip:  # horizontal flip
            img_crop = cv2.flip(img_crop,1)
            if black_white: img_crop = np.expand_dims(img_crop, 2)
        if random.uniform(0,1) < 0: # mess up contrast
            img_crop = mess_contrast(img_crop, (0.8, 1.4), (-40, 40))
        band = random.randint(0,0)  # FIXME DISABLED FOR NOW
        if band:  # add black band to bottom-right
            band_width, band_height = add_black_band(img_crop)

        # resize to input_H by input_W
        factor_Y = input_H / float(img_crop.shape[0])
        factor_X = input_W / float(img_crop.shape[1])
        img_resized = cv2.resize(img_crop, (input_W, input_H))
        if black_white:
            img_resized = img_resized.reshape(input_W, input_H, 1)
        img   = torch.from_numpy(img_resized.transpose([2,0,1])).float()
        label = torch.zeros(1, label_H, label_W).float()
        """ Generate heat label """
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 = clip(int(round((x1 - crop_x        ) * factor_X / downsample_times)), 0, label_W)
            y1 = clip(int(round((y1 - crop_y+paste_y) * factor_Y / downsample_times)), 0, label_H)
            x2 = clip(int(round((x2 - crop_x        ) * factor_X / downsample_times)), 0, label_W)
            y2 = clip(int(round((y2 - crop_y+paste_y) * factor_Y / downsample_times)), 0, label_H)
            if flip:
                x1, x2 = label_W - x2, label_W - x1
            if x1==x2 or y1==y2: continue  # out-of-crop boxes
            label[0, y1:y2, x1:x2] = 1
        if band:
            label[0, -int(round(band_height/float(downsample_times))):, :] = 0
            label[0, :,  -int(round(band_width/float(downsample_times))):] = 0
        return img, label

# do mean subtraction, BGR order
mean_bgr = torch.FloatTensor([104, 117, 124]).view(1,3,1,1)
mean_bw  = torch.FloatTensor([124]).view(1,1,1,1)

if __name__=='__main__':
    dataset  = CropsDataset(path_crops_info='/home/noid/data/huva_plate_model8_data/crops_info.pkl')
    loader   = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)

def test_data_pipeline(ds):
    """ 
    ds = iter(loader)
    Get (imgs, labels) from loader, and write them to 'imgs' folder for visualization 
    """
    imgs, labels = next(ds)
    for i in range(imgs.size(0)):
        cv2.imwrite('imgs/{}_a.jpg'.format(i), imgs[i].numpy().transpose([1,2,0]).astype('uint8'))
        cv2.imwrite('imgs/{}_b.jpg'.format(i), th_get_jet(imgs[i], labels[i]))
"""
*************************************** Model ************************************************
"""

execfile(os.path.join(file_folder, 'model.py'))

"""
*************************************** Training ************************************************
"""

def make(name):
    global model_name, model, criterion, optimizer
    if os.path.exists('{}.pth'.format(name)):
        print('Warning: {} already exits'.format(name))
    model_name = name
    model     = VGGLikeHeatRegressor(32).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = MonitoredAdam(model.parameters(), 0.001, weight_decay=0.00001)

int_report = 1
int_output = 100
def train():
    min_loss = 999
    not_min_batches = 0
    model.train()
    for epoch in xrange(num_epoch):
        g_loss = 0.0
        for batch, (imgs, labels) in enumerate(loader):
            will_report = batch % int_report == 0

            v_imgs = Variable(imgs - (mean_bw if black_white else mean_bgr).expand_as(imgs)).cuda()
            v_labels = Variable(labels).cuda()
            v_output = model(v_imgs)
            v_loss   = criterion(v_output, v_labels)

            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step(monitor_update=will_report)
            g_loss += v_loss.data[0]
            if will_report:
                avg_loss = g_loss / int_report
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    not_min_batches = 0
                else:
                    not_min_batches += 1
                print('{}, {}, {:6f} [{:6f}/{:6f}] [{}]'.format(
                    epoch, batch, g_loss/int_report, get_model_param_norm(model), optimizer.update_norm, not_min_batches))
                g_loss = 0.0
            if batch % int_output == 0:
                output = v_output.data.cpu()
                for i in range(batch_size):
                    cv2.imwrite('imgs/{}_a.jpg'.format(i), imgs[i].numpy().transpose([1,2,0]).astype('uint8'))
                    cv2.imwrite('imgs/{}_b.jpg'.format(i), th_get_jet(imgs[i], labels[i]))
                    cv2.imwrite('imgs/{}_c.jpg'.format(i), th_get_jet(imgs[i], output[i]))

def set_learning_rate(lr):
    """
    Since we only have one group, it's enough to do optimizer.param_groups[0]['lr'] = lr
    but do the proper thing anyway
    """
    for group in optimizer.param_groups:
        group['lr'] = lr


"""
*************************************** Auto-labelling ************************************************
"""

def auto_label_batch(data_batch, _model_name, mode='label'):
    """
    use model named _model_name to label an entire batch with data_batch=data_batch
    """
    assert _model_name == model_name
    folderkeys = sorted(db.folder_registry.keys())
    for folderkey in folderkeys:
        folder = db.folder_registry[folderkey]
        if folder.data_batch() != data_batch: continue
        auto_label_folder(folder, _model_name, mode)

def auto_label_folder(folder, _model_name, mode='label'):
    """
    use model named _model_name to label an entire folder
    """
    assert _model_name == model_name
    for frame in folder.frames:
        if not frame.has_run_rfcn(): continue
        auto_label_frame(frame, model_name, 0.5, mode)

def auto_label_frame(frame, model_name, threshold=0.7, mode='display', skip_existing=True):
    """
    1. load the image
    2. for each cbi, take a corresponding crop and feed through model
       - run the output through cv2.findContours then cv2.boundingRect
       - show the boundingRect is the contour area is greater than some value

       On cropping:
         instead of cropping only the tight bounding box, we can crop an extended bounding box, then remove any heat
         generated from outside the tight bounding box during contour generation. This allows us to input data in a way
         similar to training data, without getting labels from non-bb'ed cars
       
    mode is either 'display' or 'label'
    - 'display': plt.imshow the auto-detected labels
    - 'label'; directly add the detected label to corresponding cbi
    """
    model.eval()
    img = cv2.imread(frame.absolute_path())
    H, W = img.shape[0], img.shape[1]
    num_cars = 0
    num_detected = 0
    # for every car box, crop
    for i, car_bbox in enumerate(frame.parts):
        assert isinstance(car_bbox, db2.BBox)
        assert car_bbox.typename == 'car'
        x1,y1,x2,y2 = car_bbox.bbox
        """ skip all car_bbox that already have a plate label """
        plate_bboxes = car_bbox.get_typed_parts('plate')
        if skip_existing and len(plate_bboxes) > 0:
            print('{} already has {} plate_bbox'.format(car_bbox.unique_name(), len(plate_bboxes)))
            continue
        num_cars += 1
        """ Expand the bounding bbox """
        if True:
            x_inc = (x2-x1)*0.3
            y_inc = (y2-y1)*0.3
            x1_new = huva.clip(int(x1 - x_inc), 0, W)
            y1_new = huva.clip(int(y1 - y_inc), 0, H)
            x2_new = huva.clip(int(x2 + x_inc), 0, W)
            y2_new = huva.clip(int(y2 + y_inc), 0, H)
            img_crop = img[y1_new:y2_new, x1_new:x2_new]
            width_new  = x2_new - x1_new
            height_new = y2_new - y1_new
            crop_H, crop_W = img_crop.shape[0], img_crop.shape[1]
            max_side = max(crop_H, crop_W)
            """ Create heat mask """
            x1_lab = int(label_W * (float(x1 - x1_new) / max_side ))
            y1_lab = int(label_H * (float(y1 - y1_new) / max_side))
            x2_lab = int(label_W * (float(x2 - x1_new) / max_side ))
            y2_lab = int(label_H * (float(y2 - y1_new) / max_side))
            label_mask = np.zeros((label_H, label_W), np.float32)
            label_mask[y1_lab:y2_lab, x1_lab:x2_lab] = 1
        else:
            img_crop = img[y1:y2, x1:x2]
        crop_H, crop_W = img_crop.shape[0], img_crop.shape[1]
        max_side = max(crop_H, crop_W)
        factor = input_W / float(max_side)
        resized_H = int(crop_H * factor)
        resized_W = int(crop_W * factor)
        img_resized = cv2.resize(img_crop, (resized_W, resized_H))
        img_input = np.zeros((input_H, input_W, 3), np.float32)
        img_input[:resized_H, :resized_W] = img_resized
        # produce input
        if black_white:
            img_input = np.expand_dims(cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY), 2)
        imgs = torch.from_numpy(img_input.transpose([2,0,1])).contiguous().view(1,num_input_cn,input_H, input_W)
        v_imgs = Variable(imgs - (mean_bw if black_white else mean_bgr).expand_as(imgs)).cuda()
        v_outs = model(v_imgs)
        output = v_outs.data.cpu()
        out = output[0].numpy().reshape(label_H, label_W)
        # mask the non-bb region of the heat output
        out = out * label_mask
        threshed = out > threshold
        """ 
        jet = get_jet(img_input.astype(np.uint8), cv2.resize(label_mask, (img_input.shape[0], img_input.shape[1])))
        plt.imshow(jet)
        plt.show()
        """
        contours, hierarchy = cv2.findContours(threshed.astype(np.uint8), 1, 2)
        for j, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if mode=='display':
                print("area: {}".format(area))
            if area < 5:continue
            x,y,w,h = cv2.boundingRect(cnt)
            x1_det,y1_det,x2_det,y2_det = x,y,x+w,y+h
            label = torch.zeros(1, label_H, label_W)
            label[0, y1_det:y2_det, x1_det:x2_det] = 1
            if mode=='display':
                jet = th_get_jet(imgs[0], label)
                plt.imshow(jet)
                print('area: {}'.format(area))
                plt.show()
            elif mode=='count': #count stuffs
                num_detected += 1
            elif mode=='label': # label mode
                """ 
                recover x1_det and so on to the full-frame coordinate system
                1. multiply by downsample_times/factor to get back to a coordinate with origin at (x1_new, y1_new)
                2. add (x1_new, y1_new) to recover the origin at (0,0)
                """ 
                x1_rec = int(round((x1_det * downsample_times / factor) + x1_new))
                y1_rec = int(round((y1_det * downsample_times / factor) + y1_new))
                x2_rec = int(round((x2_det * downsample_times / factor) + x1_new))
                y2_rec = int(round((y2_det * downsample_times / factor) + y1_new))
                """ Now, put the bloody label on the shit """
                print(x1_det, y1_det, x2_det, y2_det)
                print(x1_rec, y1_rec, x2_rec, y2_rec)
                plate_bbox = db2.BBox(car_bbox, 'plate', (x1_rec, y1_rec, x2_rec, y2_rec))
                plate_bbox.label_type('auto')
                plate_bbox.auto_model(model_name)
                car_bbox.add_part(plate_bbox)
            else:
                assert False, 'unknown mode {}'.format(mode)
            # for every cbi we only detect on bbox, so break directly
            break
        if mode=='display':
            jet = th_get_jet(imgs[0], output[0])
            """ draw the bbox of the car """
            x1_b = int(round((x1 - x1_new)*factor))
            y1_b = int(round((y1 - y1_new)*factor))
            x2_b = int(round((x2 - x1_new)*factor))
            y2_b = int(round((y2 - y1_new)*factor))
            jet = jet.copy()
            cv2.rectangle(jet, (x1_b, y1_b), (x2_b, y2_b), (0, 0, 255))
            plt.imshow(jet)
            print('max: {}'.format(out.max()))
            plt.show()
    if mode=='count':
        return num_cars, num_detected

"""
*************************************** Visualization and evaluation *****************************************
"""

execfile(os.path.join(file_folder, 'vis_eval.py'))

"""
*************************************** Miscellaneous ************************************************
"""

execfile(os.path.join(file_folder, 'misc.py'))

"""
*************************************** Batch labelling ************************************************
"""

def make_label_batch(data_batch, folder_names, batch_name, output_folder):
    """
    From the set of folders with folder_names, 
    Pick the not-labelled car_bbox into a batch, add batch to db
    """
    batch = db2.Batch(db, 'platebatch', batch_name, output_folder)
    for folder in db.get_folders():
        if folder.data_batch() != data_batch: continue
        if folder.absolute_path.split('/')[-1] not in folder_names: continue
        print('Using folder: {}'.format(folder.absolute_path))
        for frame in folder.frames:
            for car_bbox in frame.parts:
                assert isinstance(car_bbox, db2.BBox)
                assert car_bbox.typename=='car'
                """ if it has no plate label, add it to our batch """
                if len(car_bbox.get_typed_parts('plate'))!=0: continue
                batch.add_unit(car_bbox)
    db.add_batch(batch)
    return batch

"""
1. batch=make_batch_for_xxxx(xxxx)
2. batch.output_batch()
3. batch.load_batch_label()
"""
def make_batch_for_xxxx(xxxx):
    """
    From the xxxx_frames folder, 
    pick the not-labelled car_bbox into a batch, add batch to db
    """
    folder = db.get_folder_by_lastpath('{}_frames'.format(str(xxxx)))[0]
    output_folder = os.path.join(db2.data_root, 'manual_batches', str(xxxx))
    assert not os.path.exists(output_folder)
    os.system('mkdir -p {}'.format(output_folder))
    print('batching unlabelled instances of folder {}'.format(folder.absolute_path))
    batch = db2.Batch(db, 'platebatch', '{}plates'.format(xxxx), output_folder)
    for frame in folder.frames:
        for car_bbox in frame.parts:
            assert isinstance(car_bbox, db2.BBox)
            assert car_bbox.typename=='car'
            """ if it has no plate label, add it to our batch """
            if len(car_bbox.get_typed_parts('plate'))!=0: continue
            batch.add_unit(car_bbox)
    db.add_batch(batch)
    return batch



"""
*************************************** Demo Stuffs ************************************************
"""

def show_folder_heat(folder):
    for frame in folder.frames:
        if frame.has_run_rfcn(): frame.heat_all_plates()

def feed_entire_frame(frame, target_width=204*4, mode='display'):
    """
    wrapper for feed_entire_image, for use with db2.Frame
    """
    img = cv2.imread(frame.absolute_path())
    feed_entire_image(img, target_width, mode)

def feed_entire_image(img, target_width=208*4, mode='display'):
    """
    Compute heat using an entire image at a time (instead of using car crops as in training)
    target_width: img is first scaled to have width=target_width before feeding through the model
    mode:
        'display': after heatmap computed, imshow the jet
        'get': after heatmap computed, return the jet
        'heat': after heatmap computed, return the heatmap
    """
    model.eval()
    assert target_width % 16 == 0
    H, W = img.shape[0], img.shape[1]
    factor = target_width / float(W)
    H_resize = int(round(H * factor / 16) * 16)
    W_resize = int(round(W * factor / 16) * 16)
    assert H_resize % 16 == 0
    assert W_resize % 16 == 0
    img_resized = cv2.resize(img, (W_resize, H_resize))
    if img_resized.ndim==2:
        img_resized = np.expand_dims(img_resized, 2)
    print(img_resized.shape)
    """ convert to torch format """
    imgs = torch.from_numpy(img_resized.astype(np.float32).transpose([2,0,1])).contiguous().view(1,num_input_cn, H_resize, W_resize)
    v_imgs = Variable(imgs - (mean_bw if black_white else mean_bgr).expand_as(imgs)).cuda()
    v_outs = model(v_imgs)
    output = v_outs.data.cpu()
    out = output[0][0].numpy()
    jet = get_jet(img_resized, cv2.resize(out, (W_resize, H_resize)))
    if mode=='display':
        plt.imshow(jet)
        plt.show()
    elif mode=='get':
        return jet
    elif mode=='heat':
        return img_resized, out
    else:
        assert False, 'unknown mode'

def make_entire_image_heat_for_folder(input_folder, output_folder):
    """
    Extracts all jpgs from input_folder, run them through feed_entire_image one by one, and output all the jet images
    into output_folder.
    This is for making the plate-heatmap demo.
    """
    inpath = os.path.join(db2.data_root, input_folder)
    outpath = os.path.join(db2.data_root, output_folder)
    assert os.path.exists(inpath)
    assert os.path.exists(outpath)
    infiles = glob(inpath+'/*.jpg')
    for infilename in infiles:
        img = cv2.imread(infilename)
        jet = feed_entire_image(img, 204*8, mode='get')
        outfilename = os.path.join(outpath, 'anno_'+infilename.split('/')[-1])
        cv2.imwrite(outfilename, jet)
        print(infilename)

def heat_box_entire_image(img, W_resize, output_path=None):
    """
    Heat an entire image, and from that heat compute the candidate plate bboxes
    Filter those bboxes with a min_area requirement, then return the bboxes [(x1,y1,x2,y2)]
    """
    img_resized, heat = feed_entire_image(img, W_resize, 'heat')
    threshed = heat > 0.2
    print(threshed.shape, img_resized.shape)
    contours, hierarchy = cv2.findContours(threshed.astype(np.uint8), 1, 2)
    boxes = []
    for cnt in contours:
        x,y,w,h = map(lambda i:i*downsample_times, list(cv2.boundingRect(cnt)))
        if w*h < 200: continue
        cv2.rectangle(img_resized, (x,y), (x+w, y+h), (255,0,0), 3)
        if output_path is not None:
            upsize_factor = img.shape[1] / float(W_resize)
            x1,y1,x2,y2 = map(lambda i:int(i*upsize_factor), [x,y,x+w,y+h])
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
            boxes.append([x1,y1,x2,y2])
    if output_path is not None:
        cv2.imwrite(output_path, img)
    else:
        if black_white:
            plt.imshow(img_resized[:,:,0], cmap='gray')
        else:
            plt.imshow(img_resized)
        plt.show()
    return boxes

def sequence_demo(W_resize=208*4, output=False):
    output_root = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/demos/number_sequence'
    raw_filenames = sorted(glob(output_root+'/raw/*.jpg'))
    name_to_boxes = {}
    for filename in raw_filenames:
        short_name = filename.split('/')[-1]
        if black_white:
            img = np.expand_dims(cv2.imread(filename, 0), 2)
        else:
            img = cv2.imread(filename)
        if output:
            output_path = os.path.join(output_root, 'with_plates', short_name)
            boxes = heat_box_entire_image(img, W_resize, output_path)
            name_to_boxes[short_name] = boxes
        else:
            boxes = heat_box_entire_image(img, W_resize)
        print(short_name)
    if output:
        cPickle.dump(name_to_boxes, open(os.path.join(output_root, 'with_plates', 'name_to_boxes.pkl'), 'wb'))

def load_model(name):
    global model, model_name
    if model_name == name: return
    model = torch.load(os.path.join(file_folder, '{}.pth'.format(name)))
    model_name = name
    print('loaded {}'.format(name))

"""
*************************************** Output crops ************************************************
"""

def output_bigcrops_for_training(output_folder):
    """
    Generate the training set for training localizer models.
    """
    assert os.path.exists(output_folder)
    crops_info = []
    for folder in db.get_folders():
        for frame in folder.frames:
            """ watch for both manual and auto plates when we add heat"""
            all_plates = [p for c in frame.parts for p in c.parts 
                    if p.typename=='plate' and p.label_type() in ['auto', 'manual']]
            plate_bboxes = [p.bbox for p in all_plates]
            lazy_img = LazyImage(frame.absolute_path())
            for car_bbox in frame.parts:
                if len(car_bbox.parts)==0: continue
                assert len(car_bbox.parts)==1
                plate_bbox = car_bbox.parts[0]
                assert isinstance(plate_bbox, db2.BBox)
                assert plate_bbox.typename=='plate'
                if plate_bbox.label_type() =='auto' and random.uniform(0,1) < 0.7:
                    continue
                if plate_bbox.label_type() =='none' and random.uniform(0,1) < 0.5:
                    continue
                """ 
                this sample can be used for training if it is manually labelled. Both 'manual' and 'none' are manually
                labelled. 'none' just means this crop doesn't have a plate
                """
                assert plate_bbox.label_type() in ['manual', 'none', 'auto']
                img = lazy_img.get()
                img_H = img.shape[0]
                img_W = img.shape[1]
                """ expand the bounding box and crop """
                x1,y1,x2,y2 = car_bbox.bbox
                width = x2 - x1
                height= y2 - y1
                width_inc = width / 2
                height_inc = height / 2
                x1_new = huva.clip(x1 - width_inc,  0, img_W)
                y1_new = huva.clip(y1 - height_inc, 0, img_H)
                x2_new = huva.clip(x2 + width_inc,  0, img_W)
                y2_new = huva.clip(y2 + height_inc, 0, img_H)
                width_new  = x2_new - x1_new
                height_new = y2_new - y1_new
                img_crop = img[y1_new:y2_new, x1_new: x2_new, :].copy()
                """ find all plate bbox that lay within the new crop """
                overlaps = rfcn.get_overlapping_bboxes([x1_new, y1_new, x2_new, y2_new], plate_bboxes)
                heatboxes = []
                for overlap in overlaps:
                    x1_o, y1_o, x2_o, y2_o = overlap
                    x1_o = clip(x1_o, 0, width_new)
                    y1_o = clip(y1_o, 0, height_new)
                    x2_o = clip(x2_o, 0, width_new)
                    y2_o = clip(y2_o, 0, height_new)
                    heatboxes.append((x1_o, y1_o, x2_o, y2_o))
                """ wrap up """
                save_path = os.path.join(output_folder, car_bbox.unique_name(with_jpg=True))
                cv2.imwrite(save_path, img_crop)
                crops_info.append((save_path, heatboxes))
    path_crops_info = os.path.join(output_folder, 'crops_info.pkl')
    cPickle.dump(crops_info, open(path_crops_info, 'wb'))

def get_good_crop(car_bbox, plate_bbox, img):
    """
    Takes a reasonably sized crop of the plate
    - 1-line plate: horozontally expand 0.3 randomly, vertically 1/3 of that
    - 2-line plate: vertically expand 0.3 randomly, horizontally 3x of that
    """
    assert isinstance(car_bbox, db2.BBox)
    assert isinstance(plate_bbox, db2.BBox)
    # cx,cy,cw,ch = car_bbox.xywh()
    px,py,pw,ph = plate_bbox.xywh()
    centerx = px + pw/2
    centery = py + ph/2
    # determine crop size
    H_target = make_multiple(ph * 1.5, 24)
    W_target = make_multiple(pw * 1.5, 24)
    """
    is_2line = (pw / float(ph)) < 2.4
    if is_2line: # 2 line
        H_target = make_multiple(int(ph * 1.3), 24)
        W_target = make_multiple(3 * H_target, 24)
        assert H_target >= ph
        assert W_target >= pw
    else:        # one line
        W_target = int(pw * 1.3)
        H_target = int(W_target / 3)
        assert H_target >= ph
        assert W_target >= pw
    """
    # choose a center crop
    x1 = centerx - W_target / 2
    x2 = centerx + W_target / 2
    y1 = centery - H_target / 2
    y2 = centery + H_target / 2
    x1 = clip(x1, 0, img.shape[1])
    x2 = clip(x2, 0, img.shape[1])
    y1 = clip(y1, 0, img.shape[0])
    y2 = clip(y2, 0, img.shape[0])
    crop = img[y1:y2, x1:x2]
    return crop

def output_plate_crops_for_recognizer(
            output_folder, min_area=1000, max_area=99999,
            type_set=['manual', 'auto'], 
            with_contrast_norm=False, bw=False):
    """
    output crops for plates (manual and auto) into output_folder
    filter out plates whose area < min_area
    select crop region using get_good_crop
    """
    if with_contrast_norm:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    assert os.path.exists(output_folder)
    folder_keys = sorted(db.folder_registry.keys())
    for folder_key in folder_keys:
        folder = db.folder_registry[folder_key]
        for frame in folder.frames:
            if not frame.has_run_rfcn(): continue
            lazy_img = LazyImage(frame.absolute_path())
            selected = False
            for car_bbox in frame.get_typed_parts('car'):
                for plate_bbox in car_bbox.get_typed_parts('plate'):
                    # select for manually labelled and automatically labelled plates
                    if plate_bbox.label_type() not in type_set: continue
                    # filter out tiny crops
                    x,y,w,h = plate_bbox.xywh()
                    if w*h < min_area or w*h > max_area: continue
                    # select proper crop region
                    plate_crop = get_good_crop(car_bbox, plate_bbox, lazy_img.get())
                    # color transforms
                    if bw or with_contrast_norm:
                        plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    if with_contrast_norm:
                        plate_crop = clahe.apply(plate_crop)
                    cv2.imwrite(os.path.join(output_folder, plate_bbox.unique_name(with_jpg=True)), plate_crop)
                    selected = True
            if selected: print frame.absolute_path()





if __name__ == '__main__':
    if 'test_data' in sys.argv:
        ds = iter(loader)
        test_data_pipeline(ds)
