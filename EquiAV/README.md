# Re-Implementation of EquiAV, Finetuned on Target Subset of AudioSet


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.10.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)


This folder contains a re-implementation of the EquiAV model, configured for learning audio event classification for a distributed system in the home.

## Getting Started

### API

To utilize the API for inference using the EquiAV model, first install the requirements using:

```
pip install -r requirements.txt
```

Then, build the package using:

```
pip install -e .
```

Then, in your scripts, the API functions is accessible under 'EquiAV.inference'. For example, here is a script that can be used, once metadata, class_indices_path, and weights_path have been configured for the proper dataset and fine-tuned weights:

```python
import sys
import json
import csv
import argparse

from EquiAV.inference import load_audio_model, audio_event_classification

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "InferenceArgs")

parser.add_argument('--device', type=str,   default='cpu',   help="The type of device to use.")
parser.add_argument('--modality_mode', type=str, default="audio_only", help="The modality mode (audio_only, video_only, multimodal) for the model.")
parser.add_argument('--head_type', type=str, default="linear", help="The type of layer to use for the classification head (linear, mlp)")

# Data path configurations
parser.add_argument('--metadata',type=str, default="./datasets/dataprep/Xiao_Experiments/test.json", help="Path of dataset metadata.")
parser.add_argument('--class_indices_path', type=str, default="./datasets/dataprep/Xiao_Experiments/class_labels_indices.csv", help="The path to the .csv that maps label ID to its index.")
parser.add_argument('--weights_path', type=str, default="./pretrained_weights/online_model/model/model_bestLoss_ft.pth", help="The path to the EquiAV model weights.")

# Audio parsing configurations (Recommend not touching).
parser.add_argument('--num_mel_bins', type=int, default=128, help="The number of bins for the mel spectrogram.")
parser.add_argument('--target_length', type=int, default=1024, help="The maximum length of the audio.")
parser.add_argument('--norm_mean', type=int, default=-4.346, help="The mean of the audio values for normalization.")
parser.add_argument('--norm_std', type=int, default=4.332, help="The standard deviation of the audio values for normalization.")

# Output configurations
parser.add_argument('--output_csv_path', type=str, default="../output/experiments/predictions.csv", help="The output path for the .csv file of predictions for each example.")

args = parser.parse_args()


def main():

    # ----------------------------
    # Load evaluation metadata
    # ----------------------------
    with open(args.metadata, 'r') as f:
        data = json.load(f)
        examples = data.get("data", [])
        if not examples:
            print("No examples found in JSON.")
            sys.exit(1)
    
    wav_files = {}

    for example in examples:
        id = example.get("video_id")
        wav_file_path = example.get("wav")
        wav_files[id] = wav_file_path

    # ----------------------------
    # Load the model
    # ----------------------------
    model, class_labels_df = load_audio_model(args.class_indices_path, args.weights_path, args.device, args.head_type, args.modality_mode, args.num_mel_bins)

    # ----------------------------
    # Run inference
    # ----------------------------
    results, inference_times = audio_event_classification(
        wav_files,
        model,
        class_labels_df,
        args.target_length,
        args.num_mel_bins,
        args.norm_mean,
        args.norm_std
    )

    # ----------------------------
    # Reformat results for writing to CSV
    # ----------------------------
    reformatted_results = {}
    for id in results.keys():
        reformatted_results[id] = ",".join(results[id])

    # ----------------------------
    # Save predictions to CSV
    # ----------------------------
    with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "predicted_label"])
        writer.writeheader()
        writer.writerows(reformatted_results)

    print(f"Saved predictions for {len(reformatted_results)} examples to {args.output_csv_path}")

    # ----------------------------
    # Print/save inference times
    # ----------------------------
    average_time = sum(inference_times.values()) / len(inference_times)
    print(f"Average inference time per example: {average_time:.4f} seconds")

if __name__ == '__main__':
    main()

```

For the proper weights for our fine-tuned model for audio data on household events, please download from [here](https://drive.google.com/file/d/13qiy-7yPHrcXIUGo8Kv9l8gYnR842uVU/view?usp=drive_link). For the class indices for the targeted 21 classes of household events used to fine-tune this model, please download the class indices .csv from [here](https://drive.google.com/file/d/1ZMMF3QuTK6SNzN5dx6hN5Gt9J8vCVnb2/view?usp=drive_link).


### 1. Prepare Environment & Dataset
Create conda environment
```
conda env create -f equiav.yaml
conda activate equiAV
```

Please prepare the finetuning datasets. The folder structure does not matter, but create a JSON file for each dataset to use. The format of the JSON file is as follows:

```
{
 "data": [
  {
   "video_id": "--4gqARaEJE_0_10000",
   "wav": [path to wav file],
   "labels": [label] # ex. "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
  },
  {
   "video_id": "--BfvyPmVMo_20000_30000",
   "wav": [path to wav file],
   "labels": [label] # ex. "/m/03l9g"
  },
  ...
 ]
}
```

For our purposes, the "video_id" signifies the id of the audio source, the "wav" is the path to the .wav file, and the labels are the positive labels from the targeted classes. Additionally, these JSON files should be located within the **datasets/dataprep** folder. Under this folder, subfolders should be organized by dataset as follows:

```
EquiAV/
├── datasets/
|   ├── AudioVisual.py
│   └── dataprep/
│       └── AudioSet_2M/    
│           ├── train.json
│           ├── test.json
│           ├── audioset_for_retrieval.json
│           ├── class_label_indices.csv
│           └── weights.csv 
│       └── AudioSet_20K/   
│           ├── train.json
│           ├── test.json
│           └── class_label_indices.csv
│       └── VGGSound/       
│           ├── train.json
│           ├── test.json
│           ├── VGGSound_for_retrieval.json
│           ├── class_label_indices.csv
│           └── weights.csv 
├── loss/
├── models/
├── optimizer/
├── pretrained_weights/
|   ├── mae_pretrain_vit_base.pth
...
```
Since the amount of data may vary depending on when it is downloaded, we recommend configuring a JSON file that suits your environment. The weight files, label CSV files, and the set used for retrieval follow the excellent previous work by [CAV-MAE](https://github.com/YuanGongND/cav-mae).

The pre-trained model from the original EquiAV paper can be downloaded from [here](https://docs.google.com/uc?export=download&id=1QCvBcu-CAXFLKqfk0G7niO2JO5kf74K6).

**JSON File link**

|          Dataset       |                              Content                             |
|:-------------------------:|:----------------------------------------------------------------:|
|   [AudioSet-2M](https://drive.google.com/drive/folders/1Nqz41Y-QS5FPsrgkkCIKLsL51m1HruA5?usp=drive_link)   |   (JSON) train, test, retrieval (CSV) weight, label  |
|   [AudioSet-20K](https://drive.google.com/drive/folders/1kxjzTjUR4k-68otIhzTP2UaZc9vHusR2?usp=drive_link)  |        (JSON) train, test (CSV) label       |
|   [VGGSound](https://drive.google.com/drive/folders/1rLv8fTpUNqkdQjD3T6RT5iHpGFmv1erG?usp=sharing)   |   (JSON) train, test, retrieval (CSV) weight, label   |


### 2. Fine-tuning
Use the script below to perform finetuning according to desired **dataset**(AudioSet2M, AudioSet20K, VGGSound). For the purposes of this system, we use audio_only as the mode, as our task on requires audio event classification, without any video information present.

```
python ft_main.py \
    --gpu '0,1,2,3,4,5,6,7' \
    --model 'ft_EquiAV' \
    --dataset [finetuning dataset] \
    --pretrained_model [path to EquiAV pretrained] \
    --max_epoch 50 \
    --warmup_epoch 1 \
    --batch_size 32 \
    --trainfunc_ft bceloss \
    --lr 1e-4 \
    --ftmode 'audio_only' \
    --save_path [path to save model] \
    --no_wandb
```


### 3. Inference

Use the script below to perform inference on a set of specified audio clips:


```
python inference.py \
    --model 'ft_EquiAV' \
    --weights_path [path to EquiAV weights] \
    --metadata [path to metadata of audio and video files] \
    --class_indices_path [path to class (label) name to index pairings] \
    --modality_mode 'audio_only' \
    --output_csv_path [path to save csv of inference results]
```
