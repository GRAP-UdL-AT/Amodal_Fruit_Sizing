# from typing import final
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import  GenericMask
import matplotlib.pyplot as plt
import pdb, copy


# Prepare the dataset
def get_AmodalFruitSize_dicts(root_dir, split):
    depth_path = root_dir + 'depth_maps'
    diameters_path = root_dir + 'GT_diameter.txt'
    json_file_i = os.path.join(root_dir + 'gt_json/' + split, "via_region_data_instance.json")
    json_file_a = os.path.join(root_dir + 'gt_json/' + split, "via_region_data_amodal.json")

    imgs_anns_inst = fix_AmodalFruitSize_dicts(root_dir, split)
    with open(json_file_a) as f:
        imgs_anns_amod = json.load(f)

    with open(diameters_path) as f:
        lines = f.readlines()

    list_2020 = []
    list_2018 = []
    for line in lines:
        if '2020' in line:
            list_2020.append(line)
        else:
            list_2018.append(line)

    dataset_dicts = []
    wrong_ids = []
    for idx, v in enumerate(imgs_anns_inst.values()):
        record = {}
        filename = os.path.join(root_dir + 'images/' + split, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        name = filename.split('/')[-1]
        name_amodal = name + str(v['size'])
        name = name.split('.')[0]

        # Load amodal info
        v_a = imgs_anns_amod[name_amodal]

        record["file_name"] = filename
        record["depth_file"] = os.path.join(depth_path, name + '.npy')
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        annos_a = v_a["regions"]
        objs = []
        a = 0
        for key_ in annos.keys():
            anno = annos[key_]
            if 'apple_ID' in anno['region_attributes'].keys():

                appleId = anno['region_attributes']['apple_ID']
                for key_a in annos_a.keys():
                    appleId_amod = annos_a[key_a]['region_attributes']['apple_ID']

                    if appleId == appleId_amod:

                        anno_a = annos_a[key_a]
                        sem = 1
                        apple_ID = anno['region_attributes']['apple_ID']
                        num = name.split('_')[2]
                        if num[0] == '6' or num[0] == '7':

                            # This means 2020. Format: 2020_01_322.txt,63
                            s = '2020_01_' + apple_ID.zfill(3) + '.txt'
                            corr_line = [l for l in list_2020 if s in l]

                            if corr_line == []:
                                print('2020', apple_ID)
                                print(name)
                                wrong_ids.append('2020_' + apple_ID)
                                corr_line = ['blabla,50/n']

                            n = corr_line[0].split(',')
                            n = n[-1].split('/n')[0]
                            diameter = float(n)

                        elif num[0] == '2' or num[0] == '3':
                            # This means 2018
                            s = '2018_01_' + apple_ID.zfill(3) + '.txt'
                            corr_line = [l for l in list_2018 if s in l]

                            if corr_line == []:
                                print('2018', apple_ID)
                                print(name)
                                wrong_ids.append('2018_' + apple_ID)
                                corr_line = ['blabla,50/n']
                            n = corr_line[0].split(',')
                            n = n[-1].split('/n')[0]
                            diameter = float(n)

                        anno = anno["shape_attributes"]
                        anno_a = anno_a['shape_attributes']
                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]
                        # Amodal
                        px_a = anno_a["all_points_x"]
                        py_a = anno_a["all_points_y"]
                        poly_a = [(x + 0.5, y + 0.5) for x, y in zip(px_a, py_a)]
                        poly_a = [p for x in poly_a for p in x]

                        if len(poly) > 4 and len(poly_a) > 4:
                            obj = {
                                "bbox_visible": [np.min(px), np.min(py), np.max(px), np.max(py)],
                                "bbox": [np.min(px_a), np.min(py_a), np.max(px_a), np.max(py_a)],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "visible_mask": [poly],
                                "segmentation": [poly_a],
                                "diameter": [diameter],
                                "category_id": 0,
                                "appleId": apple_ID
                            }
                            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

# Prepare the dataset
def fix_AmodalFruitSize_dicts(root_dir, split):
    json_file = os.path.join(root_dir, 'gt_json', split, "via_region_data_instance.json")

    with open(json_file) as f:
        og_imgs_anns = json.load(f)

    dataset_dicts = []
    imgs_anns = copy.deepcopy(og_imgs_anns)
    for idx, kk in enumerate(og_imgs_anns.keys()):
        v = og_imgs_anns[kk]
        v_new = imgs_anns[kk]
        record = {}
        filename = os.path.join(root_dir, 'images', split, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        annos = v["regions"]
        annos_new = v_new["regions"]
        apple_ids = []
        keys = []
        a = 0
        for key_ in annos.keys():
            anno = annos[key_]
            anno_new = annos_new[key_]
            if 'apple_ID' in anno['region_attributes'].keys():
                appleId = anno['region_attributes']['apple_ID']
                # pdb.set_trace()
                if appleId == '0':
                    annos_new.pop(key_)

                else:
                    if appleId in apple_ids:
                        # pdb.set_trace()
                        id_ap = np.where(np.array(appleId) == np.array(apple_ids))[0][0]
                        rest_x = abs(np.array(annos[keys[id_ap]]['shape_attributes']['all_points_x']) - np.array(
                            annos[key_]['shape_attributes']['all_points_x'][0]))
                        rest_y = abs(np.array(annos[keys[id_ap]]['shape_attributes']['all_points_y']) - np.array(
                            annos[key_]['shape_attributes']['all_points_y'][0]))
                        suma = rest_x + rest_y
                        id_sum = np.argmin(suma)
                        len1 = len(annos[keys[id_ap]]['shape_attributes']['all_points_x'])
                        len2 = len(annos[key_]['shape_attributes']['all_points_x'])
                        new_x = np.zeros((len1 + len2,))
                        new_x[:id_sum] = annos[keys[id_ap]]['shape_attributes']['all_points_x'][:id_sum]
                        new_x[id_sum:id_sum + len2] = annos[key_]['shape_attributes']['all_points_x']
                        new_x[id_sum + len2:] = annos[keys[id_ap]]['shape_attributes']['all_points_x'][id_sum:]
                        annos_new[keys[id_ap]]['shape_attributes']['all_points_x'] = new_x
                        new_y = np.zeros((len1 + len2,))
                        new_y[:id_sum] = annos[keys[id_ap]]['shape_attributes']['all_points_y'][:id_sum]
                        new_y[id_sum:id_sum + len2] = annos[key_]['shape_attributes']['all_points_y']
                        new_y[id_sum + len2:] = annos[keys[id_ap]]['shape_attributes']['all_points_y'][id_sum:]
                        annos_new[keys[id_ap]]['shape_attributes']['all_points_y'] = new_y
                        annos_new.pop(key_)

                        if False:
                            anno_new = annos_new[keys[id_ap]]['shape_attributes']
                            px = anno_new["all_points_x"]
                            py = anno_new["all_points_y"]
                            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                            poly = [p for x in poly for p in x]
                            Gmask = GenericMask(poly, height, width)
                            gt_mask_sing = Gmask.polygons_to_mask([poly])
                            pdb.set_trace()
                            plt.imsave(v["filename"] + key_ + '.png', gt_mask_sing)
                            # assert not anno["region_attributes"]
                    else:

                        apple_ids.append(appleId)
                        keys.append(key_)

        if len(annos_new.keys()) < 1:
            imgs_anns.pop(kk)
    return imgs_anns
