
def count_label_for_folder(folder):
    num_car = 0
    num_manual_label = 0
    num_auto_label = 0
    num_no_label = 0
    num_model = {}
    for frame in folder.frames:
        for car_bbox in frame.parts:
            num_car += 1
            plates = [p for p in car_bbox.parts if p.typename=='plate']
            auto_plates = [p for p in car_bbox.parts if p.typename=='plate' and p.label_type()=='auto']
            manual_plates = [p for p in car_bbox.parts if p.typename=='plate' and p.label_type()=='manual']
            none_plates = [p for p in car_bbox.parts if p.typename=='plate' and p.label_type()=='none']
            assert len(plates)<=1
            num_auto_label += len(auto_plates)
            num_no_label += len(none_plates)
            num_manual_label += len(manual_plates)
            if len(auto_plates)==1:
                model_name = auto_plates[0].auto_model()
                if model_name not in num_model: num_model[model_name] = 0 
                num_model[model_name] += 1
    return num_car, num_manual_label, num_auto_label, num_no_label, num_model


def count_label_for_batch(data_batch):
    num_car = 0
    num_manual_label = 0
    num_auto_label = 0
    num_no_label = 0
    for folder in db.get_folders():
        if folder.data_batch() != data_batch: continue
        _num_car, _num_manual_label, _num_auto_label, _num_no_label = count_label_for_folder(folder)
        num_car         += _num_car
        num_manual_label+= _num_manual_label
        num_auto_label  += _num_auto_label
        num_no_label    += _num_no_label
    return num_car, num_manual_label, num_auto_label, num_no_label

def count_label_for_all():
    folder_keys = sorted(db.folder_registry.keys())
    for folder_key in folder_keys:
        folder = db.folder_registry[folder_key]
        print(folder.absolute_path.split('/')[-1], folder.data_batch(), count_label_for_folder(folder))

