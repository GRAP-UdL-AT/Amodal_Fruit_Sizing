# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 10, 10


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


# run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from detectron2.data.datasets import register_coco_instances
register_coco_instances("broccoli_amodal_train", {}, "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations/annotations.json", "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations")
register_coco_instances("broccoli_amodal_val", {}, "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations/annotations.json", "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations")
register_coco_instances("broccoli_amodal_test", {}, "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations/annotations.json", "/mnt/gpid07/users/jordi.gene/amodal_segmentation/data/Broccoli_WUR/annotations/orcnn_annotations")

broccoli_amodal_train_metadata = MetadataCatalog.get("broccoli_amodal_train")
print(broccoli_amodal_train_metadata)

broccoli_amodal_val_metadata = MetadataCatalog.get("broccoli_amodal_val")
print(broccoli_amodal_val_metadata)

broccoli_amodal_test_metadata = MetadataCatalog.get("broccoli_amodal_test")
print(broccoli_amodal_test_metadata)

dataset_dicts_train = DatasetCatalog.get("broccoli_amodal_train")
dataset_dicts_val = DatasetCatalog.get("broccoli_amodal_val")
dataset_dicts_test = DatasetCatalog.get("broccoli_amodal_test")

for d in random.sample(dataset_dicts_train, 1):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=broccoli_amodal_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d, segm='segmentation')
    imshow(vis.get_image()[:, :, ::-1])

for d in random.sample(dataset_dicts_train, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=broccoli_amodal_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d, segm='visible_mask')
    imshow(vis.get_image()[:, :, ::-1])





