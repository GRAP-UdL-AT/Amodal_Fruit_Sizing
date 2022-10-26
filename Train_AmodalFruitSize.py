# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import argparse
import numpy as np
import os
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
from utils import dataset_preparation

from detectron2.engine.hooks import HookBase
#from detectron2.evaluation import inference_context
#from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
#import datetime
#import logging
from detectron2.evaluation import AmodalEvaluator
#from detectron2.data.datasets import register_coco_instances

from available_cpus import available_cpu_count


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class AmodalTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return AmodalEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            20,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')
    parser.add_argument('--num_iterations',dest='num_iterations',default=15000,help='maximum number of iterations (not epochs)')
    parser.add_argument('--checkpoint_period',dest='checkpoint_period',default=500,help='save the epoch periodically every X iterations')
    parser.add_argument('--eval_period',dest='eval_period',default=500,help='evaluate the model every X iterations (with the validation set)')
    parser.add_argument('--batch_size',dest='batch_size',default=4)
    parser.add_argument('--learing_rate',dest='learing_rate',default=0.02)
    parser.add_argument('--LR_decay',dest='weight_decay',default=0.0001)
    #parser.add_argument('--batch_size_per_image',dest='bs_per_image',default=512)
    parser.add_argument('--experiment_name',dest='experiment_name',default='trial01')
    parser.add_argument('--dataset_path',dest='dataset_path',default='./datasets/data/')
    parser.add_argument('--output_dir',dest='output_dir',default='./output/',help='name of the directory where to save the output results')
    parser.add_argument('--num_workers',dest='num_workers',default=available_cpu_count())
    parser.add_argument('--num_gpus',dest='num_gpus',default=1)
    args = parser.parse_args()
    return args

def load_dataset_dicts(dataset_path, split):
    dataset_dicts_file = os.path.join(dataset_path, split + '_dataset_dicts.npy')
    print('Loading '+split+' DATASET...')
    if not os.path.exists(dataset_dicts_file):
        print('Preparing '+split+ ' DATASET...')
        dataset_dicts = dataset_preparation.get_AmodalFruitSize_dicts(dataset_path,split)
        np.save(dataset_dicts_file,np.array(dataset_dicts))
    dataset_dicts = np.load(dataset_dicts_file,allow_pickle=True)
    DatasetCatalog.register("AmodalFruitSize_"+split, lambda d=split: dataset_dicts)
    dataset_metadata = MetadataCatalog.get("AmodalFruitSize_"+split)
    return dataset_dicts, dataset_metadata

if __name__ == '__main__':

    ## Read arguments parsed
    args = parse_args()

    max_iter          = int(args.num_iterations)
    checkpoint_period = int(args.checkpoint_period)
    eval_period       = int(args.eval_period)
    bs                = int(args.batch_size)
    lr                = float(args.learing_rate)
    lr_decay          = float(args.weight_decay)
    experiment_name   = args.experiment_name
    dataset_path      = args.dataset_path
    #bs_per_image      = int(args.bs_per_image)
    num_workers       = min(int(args.num_workers), available_cpu_count())
    num_gpus          = min(int(args.num_gpus), torch.cuda.device_count())
    output_dir        = args.output_dir
    
    # Load dataset dicts (needed by AmodalTrainer!)
    dataset_dicts_train, AmodalFruitSize_train_metadata = load_dataset_dicts(dataset_path, 'train')
    dataset_dicts_val, AmodalFruitSize_val_metadata     = load_dataset_dicts(dataset_path, 'val')

    # Set config parameters
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("AmodalFruitSize_train",)
    cfg.DATASETS.TEST  = ("AmodalFruitSize_val",)

    cfg.NUM_GPUS = num_gpus #torch.cuda.device_count()
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    # solver file settings extracted from: https://github.com/facebookresearch/Detectron/blob/master/configs/04_2018_gn_baselines/scratch_e2e_mask_rcnn_R-101-FPN_3x_gn.yaml
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.WEIGHT_DECAY = lr_decay
    cfg.SOLVER.LR_POLICY = 'steps_with_decay'
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.GAMMA = 0.1
    
    cfg.SOLVER.STEPS = (0, 7000, 11000)
    cfg.SOLVER.WARMUP_ITERS = 1000
    #cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    #cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.MAX_ITER = max_iter

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (apple)

    # https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    cfg.OUTPUT_DIR = output_dir+str(experiment_name)  # "./output/"+str(experiment_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = AmodalTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()
    print('FINISHED_TRAINING')





