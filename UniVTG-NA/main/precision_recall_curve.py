import json
import numpy as np
import matplotlib.pyplot as plt


def load_gt():
    gt_data = []
    with open('data/qvhighlights/metadata/qvhighlights_val_inc_sports.jsonl') as f:
        for line in f:
            gt_data.append(json.loads(line))
    return gt_data


def load_orig_gt():
    gt_data = []
    with open('data/qvhighlights/metadata/qvhighlights_val_sports_combined.jsonl') as f:
        for line in f:
            gt_data.append(json.loads(line))
    return gt_data


def load_preds():
    with open('preds/sports_trained_shufflednegvid_combined_preds.json') as f:
        preds = json.load(f)

    return preds


def load_orig_preds():
    with open('preds/sports_combined_preds.json') as f:
        preds = json.load(f)

    return preds


def iou(pred_start_time, pred_end_time, start_time, end_time):
    union = np.max((pred_end_time, end_time)) - np.min((pred_start_time, start_time))
    intersection = np.min((pred_end_time, end_time)) - np.max((pred_start_time, start_time))
    intersection = np.max((0, intersection))
    iou = intersection/union

    return iou


def calculate_avg_probs(preds, orig_flag):
    # calculate the avg probabilities of the different sets
    pred_probs = []
    preds = preds['preds']
    for i, pred in enumerate(preds):
        pred_windows = pred['pred_relevant_windows']
        top_pred = pred_windows[0]
        pred_prob = top_pred[2]
        pred_probs.append(pred_prob)

    if orig_flag:
        avg_pos_pred = np.mean(pred_probs[:1550])
        med_pos_pred = np.median(pred_probs[:1550])
        max_pos_pred = np.max(pred_probs[:1550])
        min_pos_pred = np.min(pred_probs[:1550])
        avg_neg_pred = np.mean(pred_probs[1550:])
        med_neg_pred = np.median(pred_probs[1550:])
        max_neg_pred = np.max(pred_probs[1550:])
        min_neg_pred = np.min(pred_probs[1550:])
    else:
        avg_pos_pred = np.mean(pred_probs[776:])
        med_pos_pred = np.median(pred_probs[776:])
        max_pos_pred = np.max(pred_probs[776:])
        min_pos_pred = np.min(pred_probs[776:])
        avg_neg_pred = np.mean(pred_probs[:776])
        med_neg_pred = np.median(pred_probs[:776])
        max_neg_pred = np.max(pred_probs[:776])
        min_neg_pred = np.min(pred_probs[:776])

    print('avg pos pred: ', avg_pos_pred)
    print('med pos pred: ', med_pos_pred)
    print('max pos pred: ', max_pos_pred)
    print('min pos pred: ', min_pos_pred)
    print('avg neg pred: ', avg_neg_pred)
    print('med neg pred: ', med_neg_pred)
    print('max neg pred: ', max_neg_pred)
    print('min neg pred: ', min_neg_pred)


def precision_recall_curve(gt_data, preds, threshold):
    # determine true pos, true neg, false pos and false neg
    preds = preds['preds']
    pos = []
    neg = []
    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []
    base_pos_above_threshold = []
    # print(len(preds))
    for i, pred in enumerate(preds):
        gt_moments = gt_data[i]['relevant_windows']
        pred_windows = pred['pred_relevant_windows']

        top_pred = pred_windows[0]
        pred_start = top_pred[0]
        pred_end = top_pred[1]
        pred_prob = top_pred[2]

        # print(gt_moments)

        gt_start = gt_moments[0][0]
        gt_end = gt_moments[0][1]

        # print(i, gt_moments)

        if pred_prob >= threshold:
            if gt_start == 0 and gt_end == 0:
                neg.append(i)
                false_pos.append(i)
                continue
            else:
                base_pos_above_threshold.append(i)
                iou_scores = []
                for gt_moment in gt_moments:
                    iou_score = iou(pred_start, pred_end, gt_moment[0], gt_moment[1])
                    iou_scores.append(iou_score)
                if np.max(iou_scores) > 0.5:
                    pos.append(i)
                    true_pos.append(i)
                else:
                    neg.append(i)
                    false_neg.append(i)


        if pred_prob < threshold:
            if gt_start == 0 and gt_end == 0:
                pos.append(i)
                true_neg.append(i)
                continue
            else:
                neg.append(i)
                false_neg.append(i)

    return pos, neg, true_pos, true_neg, false_pos, false_neg


def find_base_above_threshold(gt_data, preds, threshold):
    preds = preds['preds']
    base_pos_above_threshold = []
    for i, pred in enumerate(preds):
        gt_moments = gt_data[i]['relevant_windows']
        pred_windows = pred['pred_relevant_windows']

        top_pred = pred_windows[0]
        pred_prob = top_pred[2]

        gt_start = gt_moments[0][0]
        gt_end = gt_moments[0][1]

        if pred_prob >= threshold:
            if gt_start == 0 and gt_end == 0:
                continue
            else:
                base_pos_above_threshold.append(i)

    return base_pos_above_threshold



def calculate_prec_rec(true_pos, true_neg, false_pos, false_neg):
    true_pos_count = len(true_pos)
    true_neg_count = len(true_neg)
    false_pos_count = len(false_pos)
    false_neg_count = len(false_neg)
    print('true pos: ', true_pos_count, 'true neg: ', true_neg_count, 'false pos: ', false_pos_count, ' false neg: ',
          false_neg_count)
    precision = true_pos_count / (true_pos_count + false_pos_count)
    recall = true_pos_count / (true_pos_count + false_neg_count)

    return precision, recall


def calculate_extra_stats(true_pos, true_neg, false_pos, false_neg):
    # percentage negative rejected
    frac_neg_rejected = len(true_neg) / (len(true_neg) + len(false_pos))
    frac_pos_accepted = len(true_pos) / (len(true_pos) + len(false_neg))

    return frac_neg_rejected, frac_pos_accepted


def frac_lists(gt_data, preds):
    frac_neg_rejected_list = []
    frac_pos_accepted_list = []
    frac_base_pos_threshold_list = []

    threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

    for threshold in threshold_list:
        pos, neg, true_pos, true_neg, false_pos, false_neg = precision_recall_curve(gt_data, preds, threshold)
        frac_neg_rejected, frac_pos_accepted = calculate_extra_stats(true_pos, true_neg, false_pos, false_neg)
        frac_base_pos_threshold = len(find_base_above_threshold(gt_data, preds, threshold)) / 1550
        frac_neg_rejected_list.append(frac_neg_rejected)
        frac_pos_accepted_list.append(frac_pos_accepted)
        frac_base_pos_threshold_list.append(frac_base_pos_threshold)

    return frac_neg_rejected_list, frac_pos_accepted_list, threshold_list, frac_base_pos_threshold_list


def prec_rec_list(gt_data, preds):
    threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

    precision_list = []
    recall_list = []
    for threshold in threshold_list:
        pos, neg, true_pos, true_neg, false_pos, false_neg = precision_recall_curve(gt_data, preds, threshold)
        precision, recall = calculate_prec_rec(true_pos, true_neg, false_pos, false_neg)
        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list


def plot_curve(precision_list, recall_list):
    plt.figure(0)
    plt.plot(recall_list, precision_list, 'o')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.savefig('preds/curve_neg_trained/precision_recall.png')


def plot_frac_neg_rejected(frac_neg_rejected_list, threshold_list):
    plt.figure(1)
    plt.plot(threshold_list, frac_neg_rejected_list, 'o')
    plt.xlabel('threshold')
    plt.ylabel('frac neg rejected')
    plt.savefig('preds/curve_neg_trained/frac_neg_rejected.png')


def plot_frac_pos_accepted(frac_pos_accepted_list, threshold_list):
    plt.figure(2)
    plt.plot(threshold_list, frac_pos_accepted_list, 'o')
    plt.xlabel('threshold')
    plt.ylabel('frac pos accepted')
    plt.savefig('preds/curve_neg_trained/frac_pos_accepted.png')


def plot_frac_base_pos(frac_base_pos_threshold_list, threshold_list):
    plt.figure(3)
    plt.plot(threshold_list, frac_base_pos_threshold_list, 'o')
    plt.xlabel('threshold')
    plt.ylabel('frac pos above threshold')
    plt.savefig('preds/curve_neg_trained/frac_base_pos_threshold.png')


def plot_curve_with_orig(precision_list, recall_list, orig_precision_list, orig_recall_list):
    plt.figure(0)
    plt.plot(recall_list, precision_list, 'o-')
    plt.plot(orig_recall_list, orig_precision_list, 'o-')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.savefig('preds/curve_neg_trained_shufflednegvid/precision_recall_orig.png')


def plot_frac_neg_rejected_with_orig(frac_neg_rejected_list, orig_frac_neg_rejected_list, threshold_list):
    plt.figure(1)
    plt.plot(threshold_list, frac_neg_rejected_list, 'o-')
    plt.plot(threshold_list, orig_frac_neg_rejected_list, 'o-')
    plt.xlabel('threshold')
    plt.ylabel('frac neg rejected')
    plt.savefig('preds/curve_neg_trained_shufflednegvid/frac_neg_rejected_orig.png')


def plot_frac_pos_accepted_with_orig(frac_pos_accepted_list, orig_frac_pos_accepted_list, threshold_list):
    plt.figure(2)
    plt.plot(threshold_list, frac_pos_accepted_list, 'o-')
    plt.plot(threshold_list, orig_frac_pos_accepted_list, 'o-')
    plt.xlabel('threshold')
    plt.ylabel('frac pos accepted')
    plt.savefig('preds/curve_neg_trained_shufflednegvid/frac_pos_accepted_orig.png')


def plot_frac_base_pos_with_orig(frac_base_pos_threshold_list, orig_frac_base_pos_threshold_list, threshold_list):
    plt.figure(3)
    plt.plot(threshold_list, frac_base_pos_threshold_list, 'o-')
    plt.plot(threshold_list, orig_frac_base_pos_threshold_list, 'o-')
    plt.xlabel('threshold')
    plt.ylabel('frac pos above threshold')
    plt.savefig('preds/curve_neg_trained_shufflednegvid/frac_base_pos_threshold_orig.png')




if __name__ == "__main__":
    gt_data = load_gt()
    orig_gt_data = load_orig_gt()
    preds = load_preds()
    orig_preds = load_orig_preds()
    # pos, neg, true_pos, true_neg, false_pos, false_neg = precision_recall_curve(gt_data, preds)
    precision_list, recall_list = prec_rec_list(gt_data, preds)
    orig_precision_list, orig_recall_list = prec_rec_list(orig_gt_data, orig_preds)
    # plot_curve(precision_list, recall_list)
    plot_curve_with_orig(precision_list, recall_list, orig_precision_list, orig_recall_list)

    frac_neg_rejected_list, frac_pos_accepted_list, threshold_list, frac_base_pos_threshold_list = \
        frac_lists(gt_data, preds)
    orig_frac_neg_rejected_list, orig_frac_pos_accepted_list, threshold_list, orig_frac_base_pos_threshold_list = \
        frac_lists(orig_gt_data, orig_preds)
    # plot_frac_neg_rejected(frac_neg_rejected_list, threshold_list)
    # plot_frac_pos_accepted(frac_pos_accepted_list, threshold_list)
    # plot_frac_base_pos(frac_base_pos_threshold_list, threshold_list)

    plot_frac_neg_rejected_with_orig(frac_neg_rejected_list, orig_frac_neg_rejected_list, threshold_list)
    plot_frac_pos_accepted_with_orig(frac_pos_accepted_list, orig_frac_pos_accepted_list, threshold_list)
    plot_frac_base_pos_with_orig(frac_base_pos_threshold_list, orig_frac_base_pos_threshold_list, threshold_list)

    orig_flag = False
    calculate_avg_probs(preds, orig_flag)


