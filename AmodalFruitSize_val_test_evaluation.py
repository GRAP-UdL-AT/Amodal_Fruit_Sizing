# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
#from numpy.linalg import inv
import os
import cv2
#import random
#import csv
#import json
#from tqdm import tqdm
#import time
#from tqdm import trange


# import some miscellaneous libraries
from utils import visualize

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
#from detectron2.engine import DefaultTrainer
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#from utils import dataset_preparation, utils_diameter, utils_eval
from utils import dataset_preparation, utils_eval
import argparse

pylab.rcParams['figure.figsize'] = 10,10

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

def imshow_original_amodal_instance(img, amodal, instance):
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img[:, :, [2, 1, 0]])
    axarr[0].axis("off")
    axarr[1].imshow(amodal[:, :, [2, 1, 0]])
    axarr[1].axis("off")
    axarr[2].imshow(instance[:, :, [2, 1, 0]])
    axarr[2].axis("off")

def load_dataset_dicts(dataset_path, split):
    dataset_dicts_file = os.path.join(dataset_path, split + '_dataset_dicts.npy')
    print('Loading '+split+' DATASET...')
    if not os.path.exists(dataset_dicts_file):
        print('Preparing '+split+ ' DATASET...')
        dataset_dicts = dataset_preparation.get_AmodalFruitSize_dicts(dataset_path,split)
        np.save(dataset_dicts_file,np.array(dataset_dicts))
    dataset_dicts = np.load(dataset_dicts_file,allow_pickle=True)
    try:
        DatasetCatalog.register("AmodalFruitSize_"+split, lambda d=split: dataset_dicts)
    except:
        print("AmodalFruitSize_"+split+" is already registered!")
    dataset_metadata = MetadataCatalog.get("AmodalFruitSize_"+split)
    return dataset_dicts, dataset_metadata

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')
    parser.add_argument('--experiment_name',dest='experiment_name',default='trial01')
    parser.add_argument('--test_name',dest='test_name',default='eval_01')
    parser.add_argument('--dataset_path',dest='dataset_path',default='./datasets/data/')
    parser.add_argument('--output_dir',dest='output_dir',default='./output/',help='name of the directory where to save the output results')
    parser.add_argument('--split',dest='split',default='test')
    parser.add_argument('--weights',dest='weights',default='./output/trial01/model_0002999.pth')
    parser.add_argument('--focal_length',dest='focal_length',default=5805.34)
    parser.add_argument('--iou_thr',dest='iou_thr',default=0.5)
    parser.add_argument('--nms_thr',dest='nms_thr',default=0.1)
    parser.add_argument('--confs',dest='confs',default='0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    ## Read arguments parsed
    args = parse_args()

    experiment_name = args.experiment_name
    test_name       = args.test_name
    dataset_path    = args.dataset_path
    split           = args.split
    weights_file    = args.weights
    focal_length    = float(args.focal_length)
    iou_thr         = float(args.iou_thr)
    nms_thr         = float(args.nms_thr)
    output_dir      = args.output_dir

    dataset_dicts, AmodalFruitSize_metadata  = load_dataset_dicts(dataset_path, split)
    confidence_scores = [float(i) for i in args.confs.split(',')]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (apple)

    #cfg.OUTPUT_DIR = "./output/"+str(experiment_name)+"/"+test_name
    cfg.OUTPUT_DIR = output_dir+str(experiment_name)+"/"+test_name
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0   # set the testing threshold for this model
    #cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thr
    cfg.DATASETS.TEST = ("AmodalFruitSize_test",)

    predictor = DefaultPredictor(cfg)
    output_dir = cfg.OUTPUT_DIR

    P, R, F1, AP, MAE, MBE, MAPE, RMSE, r2 =  utils_eval.detect_measure_and_eval(predictor, dataset_dicts, confidence_scores,split,iou_thr,output_dir, focal_length)












    # ## Visualization exemple
    # img = cv2.imread(dataset_dicts[2]['file_name'])
    # outputs = predictor(img)
    # visualizer_amodal = Visualizer(img[:, :, ::-1], metadata=AmodalFruitSize_metadata, scale=0.8)
    # vis_amodal = visualizer_amodal.draw_instance_predictions(outputs["instances"].to("cpu"))
    # visualizer_instance = Visualizer(img[:, :, ::-1], metadata=AmodalFruitSize_metadata, scale=0.8)
    # vis_instance = visualizer_instance.draw_instance_predictions(outputs["instances"].to("cpu"),'pred_visible_masks')
    # imshow_original_amodal_instance(img, vis_amodal.get_image()[:, :, ::-1], vis_instance.get_image()[:, :, ::-1])



