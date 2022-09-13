#from typing import final
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2.utils.visualizer import GenericMask
import cv2



def amodal_diameter_estimation(depth_map, predictions,focal_length):
    m = 2  # parameter set to remove outliers. The outlier removal will be more restrective as m is smaller
    pred_instances = np.asarray(predictions.pred_visible_masks)
    pred_amodals = np.asarray(predictions.pred_masks)
    diameters = []
    occlusions = []
    for j in range(len(pred_instances)):
        mask_in = pred_instances[j]
        mask_am = pred_amodals[j]
        depth_in = depth_map[np.logical_and(depth_map > 0, mask_in)]
        depth_in_no_outliers = depth_in[abs(depth_in - np.mean(depth_in)) < m * np.std(depth_in)] # outlier removal
        if len(depth_in_no_outliers)>=1:
            apple_mean_depth = depth_in_no_outliers.mean()
            apple_px_area = mask_am.sum()
            k = np.sqrt(4/(np.pi*np.square(focal_length)))*1000
            diameters.append(k*apple_mean_depth * np.sqrt(apple_px_area))
        else:
            diameters.append(0)
        occlusions.append(1-mask_in.sum()/mask_am.sum())
    return diameters, occlusions


def match_mask(scores, pred_instance, pred_amodal, pred_box, diameters, occlusions , dataset_dicts, im, iter_):
    # returns pred_mask with higher intersection with the gt (performs NMS)

    used_ids_gt = []
    all_sum = []
    gts_inst_mask = []
    gts_box = []
    gts_amod_mask = []
    g_diam = []
    g_id = []
    g_occ = []
    final_pred_amodal = []
    final_box = []
    final_pred_instance = []
    final_scores = []
    final_occlusions = []
    final_diameter = []
    num_gt = len(dataset_dicts[iter_]['annotations'])

    for i in range(len(pred_instance)):

        pred_mask = pred_instance[i]
        max_sum = 0
        if num_gt >= 1:
            for j, gt_poly in enumerate(dataset_dicts[iter_]['annotations']):
                gt_poly = gt_poly['visible_mask']
                Gmask = GenericMask(gt_poly, im.shape[0], im.shape[1])
                gt_mask = Gmask.polygons_to_mask(gt_poly)
                mult = gt_mask * pred_mask
                suma = sum(sum(mult))
                if suma >= max_sum:
                    idx_m = j
                    idx_p = i
                    max_sum = suma

            final_pred_instance.append(pred_instance[idx_p])
            final_pred_amodal.append(pred_amodal[idx_p])
            final_box.append(pred_box[idx_p])
            final_diameter.append(diameters[idx_p])
            final_scores.append(scores[idx_p])
            final_occlusions.append(occlusions[idx_p])
            gt_diam = dataset_dicts[iter_]['annotations'][idx_m]['diameter']
            gt_id = dataset_dicts[iter_]['annotations'][idx_m]['appleId']
            gt_poly_inst = dataset_dicts[iter_]['annotations'][idx_m]['visible_mask']
            Gmask_inst = GenericMask(gt_poly_inst, im.shape[0], im.shape[1])
            gt_instance_sing = Gmask_inst.polygons_to_mask(gt_poly_inst)
            gt_box_sing = dataset_dicts[iter_]['annotations'][idx_m]['bbox']
            gt_poly_amod = dataset_dicts[iter_]['annotations'][idx_m]['segmentation']
            Gmask_amod = GenericMask(gt_poly_amod, im.shape[0], im.shape[1])
            gt_amodal_sing = Gmask_amod.polygons_to_mask(gt_poly_amod)
            gt_occ = 1 - gt_instance_sing.sum()/gt_amodal_sing.sum()
            if idx_m in used_ids_gt:
                a = np.where(np.array(used_ids_gt) == idx_m)[0][0]
                if all_sum[a] < max_sum:
                    gts_inst_mask[a] = None
                    used_ids_gt[a] = None
                    g_diam[a] = None
                    g_id[a] = None
                    g_occ[a] = None
                    gts_box[a] = None
                    gts_amod_mask[a] = None

                else:
                    idx_m = None
                    gt_instance_sing = None
                    gt_box_sing = None
                    gt_diam = None
                    gt_id = None
                    gt_occ = None
                    gt_amodal_sing = None

            used_ids_gt.append(idx_m)
            all_sum.append(max_sum)
            gts_inst_mask.append(gt_instance_sing)
            gts_box.append(gt_box_sing)
            g_diam.append(gt_diam)
            g_id.append(gt_id)
            g_occ.append(gt_occ)
            gts_amod_mask.append(gt_amodal_sing)
        else:
            gts_inst_mask.append(None)
            gts_box.append(None)
            g_diam.append(None)
            g_id.append(None)
            g_occ.append(None)
            gts_amod_mask.append(None)

    return num_gt, gts_inst_mask, gts_amod_mask, gts_box, g_diam, g_occ, g_id, final_pred_instance, final_pred_amodal, final_box, final_diameter, final_occlusions, final_scores



def extract_polys(mask):
    # mm,contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # eddited by JGM
    poly = {}
    if len(contours) > 1:
        # base = np.load('base_cont.npy', allow_pickle=True)
        for j, cont in enumerate(contours):
            if j > 0:
                og = np.concatenate((og, cont), axis=0)
            else:
                og = cont
        # base[0] = og
        # contours= base
        contours = (og,)
    for i, cont in enumerate(contours):
        poly[i] = {}
        poly[i]['all_points_x'] = cont[:, 0, 0]
        poly[i]['all_points_y'] = cont[:, 0, 1]

    return poly

def iou_bbox(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap