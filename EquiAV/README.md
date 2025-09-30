# Re-Implementation of EquiAV, Finetuned on Target Subset of AudioSet


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.10.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)


This folder contains a re-implementation of the EquiAV model, configured for learning audio event classification for a distributed system in the home.

## Getting Started
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
