import pdb
import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import TestOptions, setup_model
from dataset import DatasetMR, start_end_collate_mr, prepare_batch_inputs_mr
from eval.eval import eval_submission
from eval.postprocessing import PostProcessorDETR
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms
from utils.span_utils import span_cxw_to_xx

import os
import logging
import importlib

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms, opt):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    pred_dict = {}
    pred_dict['preds'] = mr_res_after_nms
    if not os.path.exists(os.path.split(opt.pred_save_path[0])[0]):
        os.makedirs(os.path.split(opt.pred_save_path[0])[0])
    with open(opt.pred_save_path[0], 'w') as f:
        json.dump(pred_dict, f)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val", "test"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug,
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms, opt=opt
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()

    if criterion:
        # if not eval_loader.dataset is None:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs_mr(batch[1], opt.device, non_blocking=opt.pin_memory)
        outputs = model(**model_inputs)
        prob = outputs["pred_logits"] # the last channel may be 1 or 2.
        try:
            pos_neg_pred_class = outputs["pred_class"]
        except:
            pos_neg_pred_class = outputs["pred_logits"][0]
        if eval_loader.dataset is not None:
            if outputs["pred_logits"].shape[-1] > 1:
                prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
            if opt.span_loss_type == "l1":
                scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
                pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)

                if opt.model_id not in ['moment_detr']: # dense regression.
                    start_spans = targets['timestamp']
                    pred_spans = start_spans + pred_spans
                    mask = targets['timestamp_mask'].bool()
                    scores[~mask] = 0
                    # if opt.eval_mode == 'v4':
                    #     _mask = targets['timestamp_window'].bool()
                    #     scores[~_mask] = 0
                _raw_saliency_scores = outputs["saliency_scores"].half()
                if opt.eval_mode == 'add':
                    # pdb.set_trace()
                    _saliency_scores = outputs["saliency_scores"].half() + prob.squeeze(-1)
                else:
                    _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)

                if opt.eval_mode == 'add_mr':
                    prob = outputs["saliency_scores"].half().unsqueeze(-1) + prob
                    scores = prob[...,0]

                saliency_scores = []
                valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
                for j in range(len(valid_vid_lengths)):
                    saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
                raw_saliency_scores = []
                for j in range(len(valid_vid_lengths)):
                    raw_saliency_scores.append(_raw_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
            else:
                bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
                pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
                # TODO use more advanced decoding method with st_ed product
                pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
                scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
                pred_spans[:, 1] += 1
                pred_spans *= opt.clip_length

            # compose predictions
            for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
                if opt.span_loss_type == "l1":
                    if opt.model_id in ['moment_detr']:
                        spans = span_cxw_to_xx(spans) * meta["duration"]
                    else:
                        spans = spans * meta["duration"]
                    spans = torch.clamp(spans, 0, meta["duration"]) # added by Kevin, since window cannot be longer than video duration.

                # (#queries, 3), [st(float), ed(float), score(float)]
                cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
                if not opt.no_sort_results:
                    cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
                cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                    pred_saliency_scores=saliency_scores[idx],
                    raw_pred_saliency_scores=raw_saliency_scores[idx],
                    pred_class=pos_neg_pred_class[idx].item()
                )
                mr_res.append(cur_query_pred)
                # print(pos_neg_pred_class[idx])

        if criterion:
            try:
                eval_loader.dataset.neg_train_path
                neg = True
            except:
                neg = False

            if not neg:
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dict["loss_overall"] = float(losses)  # for logging only
                for k, v in loss_dict.items():
                    loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))
            elif neg:
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss_dict["loss_b"] = loss_dict["loss_b"] * 0
                loss_dict["loss_g"] = loss_dict["loss_g"] * 0
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dict["neg_loss"] = float(losses)  # for logging only
                for k, v in loss_dict.items():
                    loss_meters['neg_' + k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    pred_dict = {}
    pred_dict['preds'] = mr_res

    if write_tb and criterion:
        for k, v in loss_meters.items():
            # print(k)
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
        min_w_l=2, max_w_l=150, move_window_method="left",
        # process_func_names=("clip_ts", "round_multiple")
        process_func_names=["round_multiple"]   # have added `clamp' op on line 147, thus we do not need `clip_ts' again;
    )
    # todo: are we need round_multiple?
    if opt.round_multiple > 0:
        mr_res = post_processor(mr_res)
    return mr_res, loss_meters

def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters

def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate_mr,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_dataset.data, save_submission_filename)
    return metrics, metrics_nms, eval_loss_meters, latest_file_paths

def eval_epoch_neg(model, eval_dataset, eval_neg_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate_mr,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    eval_neg_loader = DataLoader(
        eval_neg_dataset,
        collate_fn=start_end_collate_mr,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_dataset.data, save_submission_filename)

    submission_neg, eval_loss_meters_neg = get_eval_res(model, eval_neg_loader, opt, epoch_i, criterion, tb_writer)
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    save_submission_filename = save_submission_filename.replace(".jsonl", "_neg.jsonl")
    metrics_neg, metrics_nms_neg, latest_file_paths_neg = (None, None, None)

    return metrics, metrics_nms, eval_loss_meters, latest_file_paths, \
           metrics_neg, metrics_nms_neg, eval_loss_meters_neg, latest_file_paths_neg

def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    eval_dataset = DatasetMR(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        v_feat_dim=opt.v_feat_dim,
        q_feat_dim=opt.t_feat_dim,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        use_cache=opt.use_cache,
    )

    if opt.lr_warmup > 0:
        # total_steps = opt.n_epoch * len(train_dataset) // opt.bsz
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


if __name__ == '__main__':
    start_inference()
