import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ceil_to_decimal(x, base=0.1):
    if x > 0:
        rounded = base * np.ceil(x/base)
    elif x < 0:
        rounded = base * np.floor(x/base)
    elif x == 0:
        rounded = base + np.floor(x/base)

    return rounded


def ceil_to_5(x, base=5):
    if x >= 0:
        rounded = base * np.ceil(x/base)
    else:
        rounded = base * np.floor(x/base)

    return rounded


def ceil_to_25(x, base=25):
    if x >= 0:
        rounded = base * np.ceil(x/base)
    else:
        rounded = base * np.floor(x/base)

    return rounded


def calculate_iou(annotations, predictions, classes_annotations, classes_predictions):
    gtmasks = annotations.transpose(1,2,0).astype(np.uint8)
    detmasks = predictions.transpose(1,2,0).astype(np.uint8)

    gtmask_num = gtmasks.shape[-1]
    detmask_num = detmasks.shape[-1]

    IoU_matrix = np.zeros((gtmask_num,detmask_num)).astype(dtype=np.float32)

    for i in range (detmask_num):
        detclass = classes_predictions[i]
        mask = detmasks[:,:,i]*255
        maskimg = np.expand_dims(mask, axis=2) # creating a dummy alpha channel image.

        for k in range (gtmask_num):
            gtclass = classes_annotations[k]
            gtmask = gtmasks[:,:,k]*255
            gtmaskimg = np.expand_dims(gtmask, axis=2)

            intersection_area = cv2.countNonZero(cv2.bitwise_and(maskimg,gtmaskimg))
            union_area = cv2.countNonZero(cv2.bitwise_or(maskimg,gtmaskimg))

            IoU = np.divide(intersection_area,union_area)

            if detclass == gtclass:
                IoU_matrix[k,i] = IoU

    return IoU_matrix


def histogram_error(diffs, min_bin, max_bin, bin_range, digit_size, text_size):
    try:
        bins = list(np.arange(min_bin, max_bin + (bin_range/10), bin_range/10))
        counts, bins, patches = plt.hist(diffs, bins)
        plt.xticks(range(int(min_bin), int(max_bin) + int(bin_range/10), int(bin_range/10)), fontsize=digit_size)
        plt.yticks(range(0, int(np.max(counts)+10), int(np.max(counts)/10)), fontsize=digit_size)
    except:
        plt.xticks(fontsize=digit_size)
        plt.yticks(fontsize=digit_size)
    plt.grid(axis='y', alpha=0.75)
    plt.title("Diameter error from the ground truth", fontsize=text_size)
    plt.xlabel("Diameter error (mm)", fontsize=text_size)
    plt.ylabel("Frequency", fontsize=text_size)

    try:
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            if count < 10 :
                plt.annotate('n={:.0f}'.format(count), (x-3, count+2))
            elif count < 100:
                plt.annotate('n={:.0f}'.format(count), (x-4, count+2))
            else:
                plt.annotate('n={:.0f}'.format(count), (x-5, count+2))
        plt.show()
    except:
        plt.show()


def histogram_error_fixed_scale(diffs, label, min_bin, max_bin, bin_range, num_bins, digit_size, text_size, color, savename):
    bins = list(np.arange(min_bin, max_bin + (bin_range/num_bins), bin_range/num_bins))
    counts, bins, patches = plt.hist(np.clip(diffs, bins[0], bins[-2]), bins=bins, color = color)

    bins_int = [int(bin) for bin in bins]
    xlabels = np.array(bins_int).astype(str)
    xlabels[0] = xlabels[1] + '<'
    xlabels[-1] = '>' + xlabels[-2]
    plt.xticks(bins, xlabels, fontsize=digit_size)
    plt.yticks(range(0, 175, 25), fontsize=digit_size)

    plt.xlabel("Diameter error (mm)", fontsize=text_size)
    plt.ylabel("Frequency", fontsize=text_size)

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        if count < 10 :
            plt.annotate('n={:.0f}'.format(count), (x-2, count+2))
        else:
            plt.annotate('n={:.0f}'.format(count), (x-3, count+2))
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()

    return bins, counts


def counts_between(bins, counts, value):
    idxs = np.where((np.asarray(bins) >= np.multiply(value, -1)) & (np.asarray(bins) < value))
    sum_between = sum(counts[list(idxs[0])])
    sum_all = sum(counts)
    prec = (sum_between / sum_all) * 100
    
    return prec, sum_between, sum_all, value

def counts_larger(bins, counts, value):
    idxs1 = np.where((np.asarray(bins) < np.multiply(value, -1)))
    idxs2 = np.where((np.asarray(bins) > value))
    sum_larger = sum(counts[list(idxs1[0])]) + sum(counts[list(idxs2[0]-1)])
    sum_all = sum(counts)
    prec = (sum_larger / sum_all) * 100

    return prec, sum_larger, sum_all, value

def scatterplot_iou(ious, vprs, max_bin, digit_size, text_size):
    occlusion_perc =  [(1-ele)*100 for ele in vprs]
    plt.plot(occlusion_perc, ious, 'o', color='blue', alpha=0.75)
    plt.xticks(range(0, 110, 10), fontsize=digit_size)
    plt.yticks(list(np.arange(0, 1.1, 0.1)), fontsize=digit_size)
    plt.title("Amodal IoU as a function of the occlusion rate", fontsize=text_size)
    plt.xlabel("Occlusion rate (%)", fontsize=text_size)
    plt.ylabel("Amodal Intersection over Union (IoU)", fontsize=text_size)
    plt.show()


def scatterplot_occlusion(diffs, vprs, max_bin, digit_size, text_size):
    occlusion_perc =  [(1-ele)*100 for ele in vprs]
    diffs_abs =  [abs(ele) for ele in diffs]
    plt.plot(occlusion_perc, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(0, 110, 10), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the occlusion rate", fontsize=text_size)
    plt.xlabel("Occlusion rate (%)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()


def scatterplot_size(diffs, gtsizes, max_bin, digit_size, text_size):
    diffs_abs =  [abs(ele) for ele in diffs]
    plt.plot(gtsizes, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(50,275,25), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the broccoli size", fontsize=text_size)
    plt.xlabel("Ground truth size of the broccoli head (mm)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()


def boxplot_time(num_images, inference_times, digit_size, text_size):
    sns.set_style("ticks")
    df = pd.DataFrame(inference_times, columns=["time"])
    f, ax = plt.subplots(figsize=(11, 2.5))
    ax = sns.boxplot(data=df["time"], orient="h", palette="colorblind")
    plt.yticks([])
    plt.title("Image analysis times when sizing {0:.0f} broccoli heads".format(num_images), fontsize=text_size)
    plt.xticks(fontsize=digit_size)
    plt.xlabel('Image analysis time (s)', fontsize=text_size)
    plt.tight_layout()
    plt.show()