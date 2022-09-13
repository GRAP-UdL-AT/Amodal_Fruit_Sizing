import numpy as np
import cv2
import os
from utils import utils_diameter
from tqdm import trange
from sklearn import linear_model, metrics


def detect_measure_and_eval(predictor, dataset_dicts, confidence_scores,split,iou_thr,output_dir, focal_length):

    ## inference
    names = []
    scores_all = []
    D_all = []
    D_gt_all = []
    tp_all = np.array([])
    fp_all = np.array([])
    tp_count = np.zeros(len(confidence_scores))
    fp_count = np.zeros(len(confidence_scores))
    gt_num_masks = 0
    diam_err = []
    init_diam_err = np.ones(len(confidence_scores))
    dict_info_txt = {}
    save_path = output_dir + '/results_txt/' + split + '/'
    if not os.path.exists(output_dir + '/results_txt/'):
        os.mkdir( output_dir + '/results_txt/')

    for iter_ in trange(len(dataset_dicts)):
        file_name = dataset_dicts[iter_]['file_name']
        im_name = file_name.split('/')[-1]
        names.append(im_name)
        im = cv2.imread(file_name)
        depth_map = np.load(dataset_dicts[iter_]["depth_file"])
        try:
            depth_map = cv2.resize(depth_map, (np.shape(im[:, :, 0])[1], np.shape(im[:, :, 0])[0]),
                                   interpolation=cv2.INTER_AREA)
        except:
            raise Exception("COULD NOT RESIZE" + dataset_dicts['depth_file'])
        outputs = predictor(im)

        predictions = outputs["instances"].to("cpu")
        diameters, occlusions = utils_diameter.amodal_diameter_estimation(depth_map, predictions,focal_length)


        scores = predictions.scores.tolist()
        pred_instances = np.asarray(predictions.pred_visible_masks)
        pred_amodals = np.asarray(predictions.pred_masks)
        pred_boxs = predictions.pred_boxes.tensor.numpy()

        num_gt, gts_inst_mask, gts_amod_mask, gts_box, gt_diam, gt_occ, gt_id, final_pred_instance, final_pred_amodal, final_box, final_diameter, final_occlusions, final_scores = utils_diameter.match_mask(scores, pred_instances, pred_amodals, pred_boxs, diameters, occlusions , dataset_dicts, im, iter_)

        gt_num_masks += num_gt

        for j, pred_instance in enumerate(final_pred_instance):
            if pred_instance.sum()>0 and predictions.pred_masks[j].sum()>0:
                poly_txt_inst = utils_diameter.extract_polys(pred_instance)
                poly_txt_amod = utils_diameter.extract_polys(final_pred_amodal[j])
                dict_info_txt['instance_all_points_x'] = str(poly_txt_inst[0]['all_points_x']).replace('\n', '')
                dict_info_txt['instance_all_points_y'] = str(poly_txt_inst[0]['all_points_y']).replace('\n', '')
                dict_info_txt['amodal_all_points_x'] = str(poly_txt_amod[0]['all_points_x']).replace('\n', '')
                dict_info_txt['amodal_all_points_y'] = str(poly_txt_amod[0]['all_points_y']).replace('\n', '')
                dict_info_txt['conf'] = str(final_scores[j])
                dict_info_txt['diam'] = str(final_diameter[j])
                dict_info_txt['occ'] = str(final_occlusions[j])
                dict_info_txt['gt_diam'] = '0'
                dict_info_txt['id'] = '0'
                dict_info_txt['gt_occ'] = '0'
                dict_info_txt['iou'] = '0'
                if gt_id[j] != None:
                    iou = utils_diameter.iou_bbox(gts_box[j], final_box[j])
                    if iou >= iou_thr:
                        dict_info_txt['gt_diam'] = str(gt_diam[j][0])
                        dict_info_txt['id'] = gt_id[j]
                        dict_info_txt['gt_occ'] = str(gt_occ[j])
                        dict_info_txt['iou'] = str(iou)
                        error_diam = abs(gt_diam[j][0] - final_diameter[j])
                        tp_all = np.concatenate((tp_all, [1]))
                        fp_all = np.concatenate((fp_all, [0]))
                        D_all.append(final_diameter[j])
                        D_gt_all.append(gt_diam[j][0])
                        scores_all.append(final_scores[j])
                    else:
                        tp_all = np.concatenate((tp_all, [0]))
                        fp_all = np.concatenate((fp_all, [1]))
                else:
                    iou = 0
                    tp_all = np.concatenate((tp_all, [0]))
                    fp_all = np.concatenate((fp_all, [1]))


                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                with open(save_path + im_name.replace('.png', '.txt'), 'a+') as f:
                    f.write("\n")
                    f.write(
                        dict_info_txt['id'] + '|' + dict_info_txt['conf'] + '|' + dict_info_txt['diam'] + '|' +
                        dict_info_txt['gt_diam'] + '|' + dict_info_txt['occ'] + '|' + dict_info_txt[
                            'gt_occ'] + '|' + dict_info_txt['iou'] + '|' + dict_info_txt['instance_all_points_x'] + '|' + dict_info_txt[
                            'instance_all_points_y']+ '|'  + dict_info_txt['amodal_all_points_x'] + '|' + dict_info_txt[
                            'amodal_all_points_y'])

                for k, s in enumerate(confidence_scores):
                    if final_scores[j] >= s:
                        if iou >= iou_thr:
                            if init_diam_err[k]:
                                diam_err.append([])
                                init_diam_err[k] = False
                            # tp_temp[j][k] = 1.0
                            tp_count[k] += 1
                            diam_err[k].append(error_diam)
                        else:
                            # fp_temp[j][k] = 1.0
                            fp_count[k] += 1

    fp = np.cumsum(fp_all)
    tp = np.cumsum(tp_all)
    rec = tp / float(gt_num_masks)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = utils_diameter.voc_ap(rec, prec, True)
    P = np.zeros(len(confidence_scores))
    R = np.zeros(len(confidence_scores))
    F1 = np.zeros(len(confidence_scores))
    AP = np.zeros(len(confidence_scores))
    MAE1 = np.zeros(len(confidence_scores))
    MAE = np.zeros(len(confidence_scores))
    MBE = np.zeros(len(confidence_scores))
    MAPE = np.zeros(len(confidence_scores))
    RMSE = np.zeros(len(confidence_scores))
    r2 = np.zeros(len(confidence_scores))
    for k, s in enumerate(confidence_scores):
        # fp_s = np.delete(fp_all[:,k], tp_all[:,k]==fp_all[:,k])
        # tp_s = np.delete(tp_all[:,k], tp_all[:,k]==fp_all[:,k])
        # fp = np.cumsum(fp_s)
        # tp = np.cumsum(tp_s)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        AP[k] = ap
        P[k] = tp_count[k] / (tp_count[k] + fp_count[k])
        R[k] = tp_count[k] / (gt_num_masks)
        F1[k] = (2 * P[k] * R[k]) / (P[k] + R[k])
        MAE1[k] = np.mean(diam_err[k])
        D=[d for i, d in enumerate(D_all) if scores_all[i]>s]
        D_gt=[d_gt for i, d_gt in enumerate(D_gt_all) if scores_all[i]>s]
        MAE[k], MBE[k], MAPE[k], RMSE[k], r2[k] = comput_linear_regresion(D, D_gt)


    print('Confidence socres:', confidence_scores)
    print('PRECISION:', P)
    print('RECALL:', R)
    print('F1:', F1)
    print('AP:', AP)
    print('MAE:', MAE)
    print('MBE:', MBE)
    print('MAPE:', MAPE)
    print('RMSE:', RMSE)
    print('r2:', r2)

    results_lines = [ 'Conf,' + ",".join(np.char.mod('%f',confidence_scores)),
                      'P   ,' + ",".join(np.char.mod('%f',P)),
                      'R   ,' + ",".join(np.char.mod('%f',R)),
                      'F1  ,' + ",".join(np.char.mod('%f',F1)),
                      'AP  ,' + ",".join(np.char.mod('%f',AP)),
                      'MAE ,' + ",".join(np.char.mod('%f',MAE)),
                      'MBE ,' + ",".join(np.char.mod('%f',MBE)),
                      'MAPE  ,' + ",".join(np.char.mod('%f',MAPE)),
                      'RMSE  ,' + ",".join(np.char.mod('%f',RMSE)),
                      'r2 ,' + ",".join(np.char.mod('%f',r2))]
    with open(output_dir+"/"+save_path.split('/')[3]+"_P_R_F1_AP_MAE_results.csv", 'w') as f:
        f.write('\n'.join(results_lines))





    return P, R, F1, AP, MAE, MBE, MAPE, RMSE, r2

def comput_linear_regresion(D, D_gt):
    x_nan = np.array(D)
    y_nan = np.array(D_gt)
    x=x_nan[~np.isnan(x_nan)]
    y=y_nan[~np.isnan(x_nan)]
    x=x.reshape(len(x),1)
    y=y.reshape(len(y),1)
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x, y)
    m = regr.coef_
    c = regr.intercept_
    D_pred = regr.predict(x)
    MAE = metrics.mean_absolute_error(y, D_pred)
    MBE = np.mean(y-D_pred)
    MAPE = metrics.mean_absolute_percentage_error(y, D_pred)
    RMSE = metrics.mean_squared_error(y, D_pred)
    r2 = metrics.r2_score(y, D_pred)
    return MAE, MBE, MAPE, RMSE, r2