# Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval

*Accepted at WACV 2025*

[Project webpage](https://keflanagan.github.io/Moment-of-Untruth)

Under the Negative-Aware Video Moment Retrieval task, UniVTG-NA allows for the rejection of negative queries while retaining much of its moment retrieval performance. 

## Data

The negative query files are stored in the directory `dataset`

There are several jsonl files with different contents - either just the positive sentences, just the negative sentences, or both the positive and negative sentences combined. The negative sentences are assigned to specific videos for each dataset however during training the sentence assignments are shuffled.

There is also `ood_negative_sentences.jsonl` which summarises the out-of-domain sentences, detailing their category, subcategory and dataset split.

### Features

Download the Slowfast+CLIP video features and the CLIP text features [here](https://drive.google.com/drive/folders/11EWYhff_6y9f-EWv8brI7Pp-bYwZaOD7?usp=sharing)

Place and extract into `UniVTG-NA/data/{dataset}`

hdf5 files for faster dataloading may be generated with `data/create_h5py.py`

Place the hdf5 feature files (`txt_clip.hdf5`, `vid_slowfast.hdf5`, `vid_clip.hdf5`) in `UniVTG-NA/data/{dataset}/h5py`

## Preparation

Set up an environment and install dependencies from the `UniVTG-NA` directory with 

```
pip install -r requirements.txt
```

Pretrained model checkpoint can be downloaded [here](https://drive.google.com/drive/folders/1eWpuTTBRaMoV4UsEteQHAf5t4dU7uwrl)

Trained model checkpoints [here](https://drive.google.com/drive/folders/1aH6mXYrGwBuJeHbAtM8j9cHxi8W3T08j?usp=sharing)

Download the pretrained checkpoint and place it in the dir `UniVTG-NA/results/pretrained_only`

Place the trained model checkpoints in the dir `UniVTG-NA/results/mr-{dataset}/finetuned`

## Training

Set `--resume` to the path to the pretrained model in the train bash script as follows

`resume=./results/pretrained_only/model_best.ckpt`

From the `UniVTG-NA` directory, run training with
```
bash scripts/qvhl_train.sh
```

Set the `in_coef` and `out_coef` parameters in the `qvhl_train.sh` script to adjust the weighting of the in-domain and out-of-domain losses

## Inference

Set `--resume` to the path to the model checkpoint in the train bash script

`resume=./results/mr-qvhighlights/finetuned/model_best.ckpt`

Run inference with
```
bash scripts/qvhl_inference_indomain.sh
bash scripts/qvhl_inference_outofdomain.sh
```

for in-domain and out-of-domain respectively.

The path to where model predictions are saved can be set in the bash script with `--pred_save_path` and by default are stored in `preds/{dataset_name}`

Run on Charades-STA by replacing 'qvhl' in the script paths above with 'charades'

## Citation

If you use code within this repository or evaluate on the Negative-Aware Video Moment Retrieval task please cite our paper.

```
@misc{flanagan2025moment,
      title={Moment of Untruth: Dealing With Negative Queries in Video Moment Retrieval}, 
      author={Kevin Flanagan and Dima Damen and Michael Wray},
      booktitle={WACV},
      year={2025}
}
```

## Acknowledgement

This codebase is based on [UniVTG](https://github.com/showlab/UniVTG)
