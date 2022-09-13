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
import matplotlib.pyplot as plt

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


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



opt = argparse.Namespace(
    annotation_dir="/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/combined_annotations/",
    write_dir="/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations/",
    classes=["broccoli", "apple"],
    amodal_separator='_amodal',
    visible_separator='_visible',
    description='broccoli_amodal_visible',
    creator_url=None,
    contributor=None,
    version=0,
    license_url=None,
    license_id=0,
    license_name=None
    )


print(opt)
print("")

images, img_basenames, annotations, annotation_basenames = list_files(opt.annotation_dir)
print("{:d} images found!".format(len(images)))
print("{:d} annotations found!".format(len(annotations)))

activate_program = check_files(img_basenames, annotation_basenames)

print("")
print("Converting annotations...")
date_created = datetime.datetime.now()
year_created = date_created.year

## make a folder inside the writedir
writeimgdir = os.path.join(opt.write_dir, "PNGImages")

if not os.path.exists(writeimgdir):
    if not os.path.exists(opt.write_dir):
        os.mkdir(opt.write_dir)
        os.mkdir(writeimgdir)
    else:
        os.mkdir(writeimgdir)
elif os.path.exists(writeimgdir) and os.path.isdir(writeimgdir):
    shutil.rmtree(writeimgdir)
    os.mkdir(writeimgdir)

## initialize the final json file
writedata = {}
writedata['info'] = {"description": opt.description, "url": opt.creator_url, "version": str(opt.version),
                     "year": str(year_created), "contributor": opt.contributor, "date_created": str(date_created)}
writedata['licenses'] = []
writedata['licenses'].append({"url": opt.license_url, "id": opt.license_id, "name": opt.license_name})
writedata['images'] = []
writedata['type'] = "instances"
writedata['annotations'] = []
writedata['categories'] = []

for k in range(len(opt.classes)):
    superclass = opt.classes[k]
    writedata['categories'].append({"supercategory": superclass, "id": (k + 1), "name": superclass})

annotation_id = 1  ## see: https://github.com/cocodataset/cocoapi/issues/507

j=0

imgname = images[j]
img = cv2.imread(os.path.join(opt.annotation_dir, imgname))
height, width = img.shape[:2]

try:
    modTimesinceEpoc = os.path.getmtime(os.path.join(opt.annotation_dir, imgname))
    modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
    date_modified = modificationTime
except:
    date_modified = None

basename, fileext = os.path.splitext(imgname)
writename = basename + ".png"
cv2.imwrite(os.path.join(writeimgdir, writename), img)

json_name = basename.split(fileext)
jn = json_name[0] + '.json'

writedata['images'].append({
    'license': 0,
    'url': None,
    'file_name': str("PNGImages/" + writename),
    'height': height,
    'width': width,
    'date_captured': None,
    'id': j
})

jsonfile = os.path.join(opt.annotation_dir, jn)
classnames = opt.classes
amodal_sep = opt.amodal_separator
visible_sep = opt.visible_separator


######### def process_json(jsonfile, classnames, amodal_sep, visible_sep): ############################
group_ids = []

with open(jsonfile, 'r') as json_file:
    data = json.load(json_file)
    for p in data['shapes']:
        group_ids.append(p['group_id'])

unique_group_ids = list(set(group_ids))
unique_group_ids.sort()

category_ids = []
amodal_masks = []
visible_masks = []
areas = []
boxes = []
crowd_ids = []


for i in range(len(unique_group_ids)):
    category_ids.append([])
    amodal_masks.append([])
    visible_masks.append([])
    areas.append([])
    boxes.append([])
    crowd_ids.append([])

################# for p in data['shapes']:###########
ps = data['shapes']
p = ps[0]

group_id = p['group_id']
amodal = False
visible = False

if amodal_sep in p['label']:
    classname = p['label'].split(amodal_sep)[0]
    amodal = True

if visible_sep in p['label']:
    classname = p['label'].split(visible_sep)[0]
    visible = True

fill_id = int(np.where(np.asarray(unique_group_ids) == group_id)[0][0])

try:
    category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
    category_ids[fill_id] = category_id
    run_further = True
except:
    print("Cannot find the class name (please check the amodal/visible separator)")
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

        cx = (orig_x2 - orig_x1) ** 2
        cy = (orig_y2 - orig_y1) ** 2
        radius = math.sqrt(cx + cy)

        circle_polygon = []

        for k in range(0, len(bearing_angles) - 1):
            ad1 = math.radians(bearing_angles[k])
            x1 = radius * math.cos(ad1)
            y1 = radius * math.sin(ad1)
            circle_polygon.append((orig_x1 + x1, orig_y1 + y1))

            ad2 = math.radians(bearing_angles[k + 1])
            x2 = radius * math.cos(ad2)
            y2 = radius * math.sin(ad2)
            circle_polygon.append((orig_x1 + x2, orig_y1 + y2))

        pts = np.asarray(circle_polygon).astype(np.float32)
        pts = pts.reshape((-1, 1, 2))
        points = np.asarray(pts).flatten().tolist()

    if p['shape_type'] == "rectangle":
        (x1, y1), (x2, y2) = p['points']
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        points = [x1, y1, x2, y1, x2, y2, x1, y2]

    if p['shape_type'] == "polygon":
        points = p['points']
        pts = np.asarray(points).astype(np.float32).reshape(-1, 1, 2)
        points = np.asarray(pts).flatten().tolist()

    if amodal:
        amodal_masks[fill_id].append(points)

        bbx, bby, bbw, bbh = cv2.boundingRect(pts)
        area = bbw * bbh
        areas[fill_id] = area
        boxes[fill_id] = [bbx, bby, bbw, bbh]
    elif visible:
        visible_masks[fill_id].append(points)

    ## labelme version 4.5.6 does not have a crowd_id, so fill it with zeros
    crowd_ids[fill_id] = 0
    status = "successful"
else:
    status = "unsuccessful"
