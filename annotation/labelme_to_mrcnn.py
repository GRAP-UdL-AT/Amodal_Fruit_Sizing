import argparse
import os
import numpy as np
import shutil
import cv2
import json
import math
import datetime
import time
from tqdm import tqdm

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")


def list_files(annotdir):
    if os.path.isdir(annotdir):
        all_files = os.listdir(annotdir)
        images = [x for x in all_files if x.lower().endswith(supported_cv2_formats)]
        annotations = [x for x in all_files if ".json" in x]
        images.sort()
        annotations.sort()

        img_basenames = [os.path.splitext(img)[0] for img in images]
        annotation_basenames = [os.path.splitext(annot)[0] for annot in annotations]

    return images, img_basenames, annotations, annotation_basenames


def check_files(img_basenames, annotation_basenames):
    activate_program = True
    diff_img_annot = list(set(img_basenames) - set(annotation_basenames))
    diff_annot_img = list(set(annotation_basenames) - set(img_basenames))

    if len(diff_img_annot) > 0:
        activate_program = False
        print("These images do not have an annotation:")
        for i in range(len(diff_img_annot)):
            print(diff_img_annot[i])
            
    print("")
    if len(diff_annot_img) > 0:
        activate_program = False
        print("These annotations do not have an image:")
        for i in range(len(diff_annot_img)):
            print(diff_annot_img[i])

    return activate_program


def process_json(jsonfile, classnames):
    group_ids = []

    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)
        for p in data['shapes']:
            group_ids.append(p['group_id'])

    only_group_ids = [x for x in group_ids if x is not None]
    unique_group_ids = list(set(only_group_ids))
    no_group_ids = sum(x is None for x in group_ids)
    total_masks = len(unique_group_ids) + no_group_ids

    all_unique_masks = np.zeros(total_masks, dtype = object)

    if len(unique_group_ids) > 0:
        unique_group_ids.sort()

        for k in range(len(unique_group_ids)):
            unique_group_id = unique_group_ids[k]
            all_unique_masks[k] = unique_group_id

        for h in range(no_group_ids):
            all_unique_masks[len(unique_group_ids) + h] = "None" + str(h+1)
    else:
        for h in range(no_group_ids):
            all_unique_masks[h] = "None" + str(h+1)    

    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    none_counter = 0 

    for p in data['shapes']:
        group_id = p['group_id']

        if group_id is None:
            none_counter = none_counter + 1
            fill_id = int(np.where(np.asarray(all_unique_masks) == (str(group_id) + str(none_counter)))[0][0])
        else:
            fill_id = int(np.where(np.asarray(all_unique_masks) == group_id)[0][0])

        classname = p['label']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if p['shape_type'] == "circle":
                # https://github.com/wkentaro/labelme/issues/537
                bearing_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]
                            
                orig_x1 = p['points'][0][0]
                orig_y1 = p['points'][0][1]

                orig_x2 = p['points'][1][0]
                orig_y2 = p['points'][1][1]

                cx = (orig_x2 - orig_x1)**2
                cy = (orig_y2 - orig_y1)**2
                radius = math.sqrt(cx + cy)

                circle_polygon = []
            
                for k in range(0, len(bearing_angles) - 1):
                    ad1 = math.radians(bearing_angles[k])
                    x1 = radius * math.cos(ad1)
                    y1 = radius * math.sin(ad1)
                    circle_polygon.append( (orig_x1 + x1, orig_y1 + y1) )

                    ad2 = math.radians(bearing_angles[k+1])
                    x2 = radius * math.cos(ad2)  
                    y2 = radius * math.sin(ad2)
                    circle_polygon.append( (orig_x1 + x2, orig_y1 + y2) )

                pts = np.asarray(circle_polygon).astype(np.float32)
                pts = pts.reshape((-1,1,2))
                points = np.asarray(pts).flatten().tolist()
                
            if p['shape_type'] == "rectangle":
                (x1, y1), (x2, y2) = p['points']
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]

            if p['shape_type'] == "polygon":
                points = p['points']
                pts = np.asarray(points).astype(np.float32).reshape(-1,1,2)   
                points = np.asarray(pts).flatten().tolist()

            masks[fill_id].append(points)

            ## labelme version 4.5.6 does not have a crowd_id, so fill it with zeros
            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

    return category_ids, masks, crowd_ids, status


def bounding_box(masks):
    areas = []
    boxes = []

    for _ in range(len(masks)):
        areas.append([])
        boxes.append([])


    for i in range(len(masks)):
        points = masks[i]
        all_points = np.concatenate(points)

        pts = np.asarray(all_points).astype(np.float32).reshape(-1,1,2)
        bbx,bby,bbw,bbh = cv2.boundingRect(pts)

        area = bbw*bbh 
        areas[i] = area                      
        boxes[i] = [bbx,bby,bbw,bbh]

    return areas, boxes


def visualize(img, category_ids, masks, boxes, classes):
    colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    color_list = np.remainder(np.arange(len(classes)), len(colors))
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 1
    thickness = 3
    text_color1 = [255, 255, 255]
    text_color2 = [0, 0, 0]

    img_vis = img.copy()

    for i in range(len(masks)):
        points = masks[i]
        bbx,bby,bbw,bbh = boxes[i]
        category_id = category_ids[i]
        class_id = category_id-1
        _class = classes[class_id]
        color = colors[color_list[class_id]]

        for j in range(len(points)):
            point_set = points[j]
            pntset = np.asarray(point_set).astype(np.int32).reshape(-1,1,2) 
            img_vis = cv2.polylines(img_vis, [pntset], True, color, thickness)

        img_vis = cv2.rectangle(img_vis, (bbx, bby), ((bbx+bbw), (bby+bbh)), color, thickness)

        text_str = "{:s}".format(_class)
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        if bby < 100:
            text_pt = (bbx, bby+bbh)
        else:
            text_pt = (bbx, bby)

        img_vis = cv2.rectangle(img_vis, (text_pt[0], text_pt[1] + 7), (text_pt[0] + text_w, text_pt[1] - text_h - 7), text_color1, -1)
        img_vis = cv2.putText(img_vis, text_str, (text_pt[0], text_pt[1]), font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

    return img_vis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_dir', type=str, default='Data/annotations/mrcnn_annotations', help='file directory with the images and the annotations')
    parser.add_argument('--write_dir', type=str, default='Data/annotations/mrcnn_annotations', help='file directory to write the images and the annotations')
    parser.add_argument('--visualize_dir', type=str, default=[], help='file directory to write the visualizations of the annotations')
    parser.add_argument('--classes', nargs="+", default=[], help='a list with all the classes')
    parser.add_argument('--description', type=str, default='broccoli_amodal_visible', help='description of the dataset')
    parser.add_argument('--creator_url', type=str, default=None, help='url of the creator')
    parser.add_argument('--contributor', type=str, default=None, help='contributor')
    parser.add_argument('--version', type=int, default=0, help='version id')
    parser.add_argument('--license_url', type=str, default=None, help='url of the license')
    parser.add_argument('--license_id', type=int, default=0, help='license id')
    parser.add_argument('--license_name', type=str, default=None, help='name of the license')
    
    opt = parser.parse_args()
    print(opt)
    print("")

    images, img_basenames, annotations, annotation_basenames = list_files(opt.annotation_dir)
    print("{:d} images found!".format(len(images)))
    print("{:d} annotations found!".format(len(annotations)))

    activate_program = check_files(img_basenames, annotation_basenames)

    if activate_program:
        print("")
        print("Converting annotations...")
        date_created = datetime.datetime.now()
        year_created = date_created.year

        ## make a folder inside the writedir
        writeimgdir = os.path.join(opt.write_dir,"PNGImages")

        if not os.path.exists(writeimgdir):
            if not os.path.exists(opt.write_dir):
                os.mkdir(opt.write_dir)
                os.mkdir(writeimgdir)
            else:
                os.mkdir(writeimgdir)
        elif os.path.exists(writeimgdir) and os.path.isdir(writeimgdir):
            shutil.rmtree(writeimgdir)
            os.mkdir(writeimgdir) 

        ## make the visualization folder
        if opt.visualize_dir:
            if not os.path.exists(opt.visualize_dir):
                os.mkdir(opt.visualize_dir)

        ## initialize the final json file
        writedata = {}
        writedata['info'] = {"description": opt.description, "url": opt.creator_url, "version": str(opt.version), "year": str(year_created), "contributor": opt.contributor, "date_created": str(date_created)}
        writedata['licenses'] = []
        writedata['licenses'].append({"url": opt.license_url, "id": opt.license_id, "name": opt.license_name})
        writedata['images'] = []
        writedata['type'] = "instances"
        writedata['annotations'] = []
        writedata['categories'] = []

        for k in range(len(opt.classes)):
            superclass = opt.classes[k]
            writedata['categories'].append({"supercategory": superclass, "id": (k+1), "name": superclass})

        annotation_id = 1   ## see: https://github.com/cocodataset/cocoapi/issues/507

        for j in tqdm(range (len(images))):
            imgname = images[j]
            readdir = opt.annotation_dir
            img = cv2.imread(os.path.join(readdir, imgname))

            height, width = img.shape[:2]

            try:
                modTimesinceEpoc = os.path.getmtime(os.path.join(readdir, imgname))
                modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
                date_modified = modificationTime
            except:
                date_modified = None

            basename, fileext = os.path.splitext(imgname)
            writename = basename + ".png"
            cv2.imwrite(os.path.join(writeimgdir,writename),img)

            json_name = basename.split(fileext)
            jn = json_name[0]+'.json'
            
            writedata['images'].append({
                            'license': 0,
                            'url': None,
                            'file_name': str("PNGImages/" + writename),
                            'height': height,
                            'width': width,
                            'date_captured': None,
                            'id': j
                        })
       
            # Procedure to store the annotations in the final JSON file
            category_ids, masks, crowd_ids, status = process_json(os.path.join(readdir, jn), opt.classes)
            areas, boxes = bounding_box(masks)

            if opt.visualize_dir:
                img_vis = visualize(img, category_ids, masks, boxes, opt.classes)
                cv2.imwrite(os.path.join(opt.visualize_dir, writename), img_vis)

            for q in range(len(category_ids)):
                category_id = category_ids[q]
                mask = masks[q]
                bb_area = areas[q]
                bbpoints = boxes[q]
                crowd_id = crowd_ids[q]

                writedata['annotations'].append({
                        'id': annotation_id,
                        'image_id': j,
                        'category_id': category_id,
                        'segmentation': mask,
                        'area': bb_area,
                        'bbox': bbpoints,
                        'iscrowd': crowd_id
                    })
        
                annotation_id = annotation_id+1
                

        with open(os.path.join(opt.write_dir,"annotations.json"), 'w') as outfile:
            json.dump(writedata, outfile)

        status == "successful"
    else:
        print("")
        print("Please update your images/annotations first!")


print("")
if status == "successful":
    print("Conversion finished!")
else:
    print("Unfortunately, the conversion could not be done!")
