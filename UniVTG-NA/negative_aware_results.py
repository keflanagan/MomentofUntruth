import json
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gt_path", default="", type=str, help="path to the ground truth file")
parser.add_argument("--pred_path", default="", type=str, help="path to the predictions file")
args = parser.parse_args()


def load_pred(pred_file):
    with open(pred_file) as f:
        pred_data = json.load(f)
    return pred_data


def load_gt(gt_path):
    gt_data = []
    with open(gt_path) as f:
        for line in f:
            gt_data.append(json.loads(line))
    return gt_data


def rejection_accuracy(pred_data, pred_threshold):
    neg_above = []
    pos_below = []
    neg_count = 0
    pos_count = 0
    for idx, pred in enumerate(pred_data['preds']):
        gt_seg = gt_data[idx]["relevant_windows"]
        pred_class = pred["pred_class"]
        if gt_seg == [[0, 0]]:
            neg_count += 1
            pos = False
            neg = True
        else:
            pos_count += 1
            pos = True
            neg = False

        if pred_class >= pred_threshold:
            if neg:
                neg_above.append(idx)
        if pred_class < pred_threshold:
            if pos:
                pos_below.append(idx)

    print('Rejection Accuracy:', round(100 * ((neg_count - len(neg_above)) / neg_count), 2))


def recall_scores(pred_data, gt_data, pred_threshold):
    "Calculate the R@1,IoU@k scores while excluding those predicted as negatives"
    true_pos30 = []
    true_pos50 = []
    true_pos70 = []
    pos_count = 0
    for i, pred in enumerate(pred_data['preds']):
        gt_moments = gt_data[i]['relevant_windows']
        pred_windows = pred['pred_relevant_windows']
        pred_class = pred["pred_class"]

        top_pred = pred_windows[0]
        pred_start = top_pred[0]
        pred_end = top_pred[1]

        gt_start = gt_moments[0][0]
        gt_end = gt_moments[0][1]

        if pred_class >= pred_threshold:
            if gt_start == 0 and gt_end == 0:
                continue
            else:
                pos_count += 1
                iou_scores = []
                for gt_moment in gt_moments:
                    iou_score = iou(pred_start, pred_end, gt_moment[0], gt_moment[1])
                    iou_scores.append(iou_score)
                if np.max(iou_scores) > 0.3:
                    true_pos30.append(i)
                if np.max(iou_scores) > 0.5:
                    true_pos50.append(i)
                if np.max(iou_scores) > 0.7:
                    true_pos70.append(i)

        if pred_class < pred_threshold:
            if gt_start == 0 and gt_end == 0:
                continue
            else:
                pos_count += 1

    recall30 = len(true_pos30) / pos_count
    recall50 = len(true_pos50) / pos_count
    recall70 = len(true_pos70) / pos_count
    print('Recall@1, IoU@0.3', round(recall30 * 100, 2))
    print('Recall@1, IoU@0.5', round(recall50 * 100, 2))
    print('Recall@1, IoU@0.7', round(recall70 * 100, 2))


def iou(pred_start_time, pred_end_time, start_time, end_time):
    union = np.max((pred_end_time, end_time)) - np.min((pred_start_time, start_time))
    intersection = np.min((pred_end_time, end_time)) - np.max((pred_start_time, start_time))
    intersection = np.max((0, intersection))
    iou = intersection/union

    return iou


if __name__ == "__main__":
    pred_data = load_pred(args.pred_path)
    gt_data = load_gt(args.gt_path)
    pred_threshold = 0.5
    rejection_accuracy(pred_data, pred_threshold)
    recall_scores(pred_data, gt_data, pred_threshold)
