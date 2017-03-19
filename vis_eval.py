
def visualize_model_plates(folder, automodel):
    """
    For the plates already automatically detected by some model and registered in the database, visualize them.
    But only visualize the subset that are labelled by model with name = automodel
    This is for debugging the performance of specific models
    """
    for frame in folder.frames:
        lazy_img = LazyImage(frame.absolute_path())
        for car_bbox in frame.parts:
            x1_c, y1_c, _, _ = car_bbox.bbox
            for plate_bbox in car_bbox.parts:
                if plate_bbox.label_type() == 'auto' and plate_bbox.auto_model()==automodel:
                    x1_p,y1_p,x2_p,y2_p = plate_bbox.bbox
                    img_frame = lazy_img.get()
                    img_crop = car_bbox.get_crop(img_frame)
                    x1 = (x1_p - x1_c)
                    y1 = (y1_p - y1_c)
                    x2 = (x2_p - x1_c)
                    y2 = (y2_p - y1_c)
                    img_crop[y1:y2, x1:x2] = [0,0,255]
                    plt.imshow(img_crop)
                    plt.show()

def visualize_none_plates(folder):
    """
    Visualize car_bbox inside this folder that does not have a plate label
    """
    for frame in folder.frames:
        lazy_img = LazyImage(frame.absolute_path())
        for car_bbox in frame.parts:
            x1_c, y1_c, _, _ = car_bbox.bbox
            if len(car_bbox.parts) == 0:
                img_frame = lazy_img.get()
                img_crop = car_bbox.get_crop(img_frame)
                plt.imshow(img_crop)
                plt.show()

def visualize_auto_plates(folder):
    """
    Show every auto-labelled bbox with plate_bbox inside the folder
    """
    for frame in folder.frames:
        lazy_img = LazyImage(frame.absolute_path())
        for car_bbox in frame.parts:
            auto_plates = [p for p in car_bbox.parts if p.typename=='plate' and p.label_type()=='auto']
            if len(auto_plates)==0: continue
            assert len(auto_plates)==1
            plate_bbox = auto_plates[0]
            x1_c,y1_c,x2_c,y2_c = car_bbox.bbox
            x1_p,y1_p,x2_p,y2_p = plate_bbox.bbox
            img_crop = lazy_img.get()[y1_c:y2_c, x1_c:x2_c]
            label = np.zeros((y2_c - y1_c, x2_c - x1_c), np.float32)
            x1_l = x1_p - x1_c
            y1_l = y1_p - y1_c
            x2_l = x2_p - x1_c
            y2_l = y2_p - y1_c
            label[y1_l:y2_l, x1_l:x2_l] = 1
            jet = get_jet(img_crop, label)
            plt.imshow(jet)
            plt.show()

def visualize_auto_plates_for_batch(data_batch):
    for folder in db.get_folders():
        if folder.data_batch() != data_batch: continue
        visualize_auto_plates(folder)


def evaluate_current_model_on_folder(folder):
    """
    Use the currently loaded model to detect plates of folder, and report the counts as
    (number of car_bbox, number of plates detected)
    """
    num_cars = 0
    num_detected = 0
    for frame in folder.frames:
        if not frame.has_run_rfcn(): continue
        cars, detected = auto_label_frame(frame, '', 0.5, mode='count', skip_existing=False)
        num_cars += cars
        num_detected += detected
    return num_cars, num_detected

def evaluate_model_on_folder(model_name, folder):
    load_model(model_name)
    return evaluate_current_model_on_folder(folder)

def evaluate_model_on_batch(model_name, data_batch):
    """
    Evaluate performance of named model on a batch of data
    Return (number of car_bbox, number of detected plates)

    With new 0.3 W-extension:
        model8,batch2: 18441, 12407
        model6,batch2: 18441, 11784
        model5,batch2: 18441, 11206
        model4,batch2: 18441, 10428
        model3,batch2: 18441, 
    """
    num_cars = 0
    num_detected = 0
    for folder in db.get_folders():
        if folder.data_batch() != data_batch:continue
        cars, detected = evaluate_model_on_folder(model_name, folder)
        print('{}: {} {}'.format(folder.absolute_path.split('/')[-1], cars, detected))
        num_cars += cars
        num_detected += detected
    return num_cars, num_detected
