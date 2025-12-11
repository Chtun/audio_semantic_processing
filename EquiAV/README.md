# Re-Implementation of EquiAV, Finetuned on Target Subset of AudioSet


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.10.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)


This folder contains a re-implementation of the EquiAV model, configured for learning audio event classification for a distributed system in the home.

## Getting Started

### API

To utilize the API for the EquiAV model inference and fine-tuning, first install the requirements using:

```
pip install -r requirements.txt
```

or, alternatively, create the conda environment:

```
conda env create -f equiav.yaml
conda activate equiAV
```

Then, build the package using:

```
pip install -e .
```

Then, in your scripts, the API functions is accessible under 'EquiAV.inference' and 'EquiAV'.ft_trainer'.

For the proper weights for our fine-tuned model for audio data on household events, please download from [here](https://drive.google.com/file/d/13qiy-7yPHrcXIUGo8Kv9l8gYnR842uVU/view?usp=drive_link). For the class indices for the targeted 21 classes of household events used to fine-tune this model, please download the class indices .csv from [here](https://drive.google.com/file/d/1ZMMF3QuTK6SNzN5dx6hN5Gt9J8vCVnb2/view?usp=drive_link).

#### Inference

For inference, here is an example script that can be used, once metadata, class_indices_path, and weights_path have been configured for the proper dataset and fine-tuned weights:

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

#### Fine-tuning

To utilize the API for continual learning via online fine-tuning using a batch size of 1 (single pair), the following script can be used for audio-only finetuning, once metadata, class_indices, and pretrained_model arguments have been configured to their proper path locations. Additionally, if you would like to extend the number of classes that may be predicted for, utilize old_class_indices to define the old class label to index mapping and class_indices to define the new class label to index mapping. Importantly, the labels shared by the old mapping and new mapping should have the same indices to take advantage of the input model. Learning rates and other hyperparameters should be configured as desired.

```python

import argparse
import warnings
import torch
import distutils
import json
import pandas as pd

from datasets.AudioVisual import format_model_input, format_label_data
from ft_trainer import *

warnings.filterwarnings(action='ignore')


## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "TrainArgs")

parser.add_argument('--device', type=str,   default='cpu',   help='The type of device to use')
parser.add_argument('--gpu', type=int,   default=0,   help='gpu id to use')

# Data definition
parser.add_argument('--dataset', type=str, default="AudioSet_20K_Targeted", help="The type of datset being used.")
parser.add_argument('--metadata',type=str, default="./datasets/dataprep/AudioSet_20K_Targeted/train.json", help='Path of dataset metadata.')
parser.add_argument('--class_indices',type=str, default="./datasets/dataprep/AudioSet_20K_Targeted/class_labels_indices.csv", help='Path of the  current dataset class index mapping.')
parser.add_argument('--old_class_indices', type=str, default='DEFAULT', help="Path of prior class index mapping for old model, if using new mapping.")
parser.add_argument('--fold',type=str, default="1", help='name of dataset definition')
parser.add_argument("--bal", type=lambda x:bool(distutils.util.strtobool(x)),  default=False, help="weight sampling for class balance ex) 'bal'")

parser.add_argument("--num_mel_bins", type=int, default=128,    help="number of mel bins of spectrogram")

# dataset augmentations for finetuning
parser.add_argument("--mixup", type=float, default=0.0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--noise", type=lambda x:bool(distutils.util.strtobool(x)),  default=False, help='if use balance sampling')
parser.add_argument('--ft_freqm', type=int, default=48, help='frequency mask max length')
parser.add_argument('--ft_timem', type=int, default=192, help='time mask max length')
parser.add_argument('--label_smooth', type=float, default=0.0, help='label smoothing')

# Data loader details
parser.add_argument('--batch_size', type=int,   default=1,    help='Batch size, number of speakers per batch')
parser.add_argument('--nDataLoaderThread', type=int, default=8,     help='Number of loader threads')
parser.add_argument('--checkloader', dest='checkloader', action='store_true', help='check the dataloders')

# Training details
parser.add_argument('--max_epoch', type=int,    default=5,          help='Maximum number of epochs')
parser.add_argument('--trainfunc_ft', type=str,    default="bceloss",   help='Finetuning loss function')

# Model definition
parser.add_argument('--model', type=str,   default="ft_EquiAV",   help='Name of model definition')
parser.add_argument('--inter_linear',     type=bool,  default=True,      help='Use the linear head for extracting invariant representation')
parser.add_argument('--head_type',     type=str,  default='linear', choices=['linear', 'mlp'],      help='Head type (linear or mlp)')
parser.add_argument('--head_dim',     type=int,  default=512,  help='Dimension for mlp hidden layer')
parser.add_argument('--aug_size_a', type=int,   default=21,         help='Dimension for data augmentation parameters')
parser.add_argument('--aug_size_v', type=int,   default=17,         help='Dimension for data augmentation parameters')
parser.add_argument("--drop_path", type=float, default=0.1,    help="drop_path value of the finetuning model")
parser.add_argument("--drop_out", type=float, default=0,    help="drop_out value of the finetuning model")

parser.add_argument('--freeze_base', type=bool,  default=True,       help='Freeze base network without MLP during training')
parser.add_argument("--ftmode", type=str, default='audio_only', help="how to fine-tune the model")
parser.add_argument('--data_aug',      type=lambda x:bool(distutils.util.strtobool(x)),  default=False,  help='Enable data_aug')


# Optimizer details
parser.add_argument('--optimizer', type=str,   default="adamw", help='sgd or adam')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay in the optimizer')

# Learning rate details
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument("--head_lr", type=float, default=1e-4, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument("--start_lr", type=float, default=2e-7, help="start point of learning rate")
parser.add_argument("--final_lr", type=float, default=1e-6, help="final point of learning rate")

# Scheduler
parser.add_argument('--scheduler',      type=str,   default="warmupcos", help='Learning rate scheduler')
parser.add_argument('--warmup_epoch',      type=int,   default=3, help='warmup epoch for cosine lr scheduler')

## Load and save
parser.add_argument('--save_path', type=str, default="./pretrained_weights/online_model", help='Path for model and logs')
parser.add_argument('--model_save_freq',     type=int, default=2, help='Frequency of saving model weight')

parser.add_argument('--pretrained_model', type=str, default="./pretrained_weights/online_model/model/model_bestLoss_ft.pth", help='pretrained model weights for finetuning')

# Accelerate training
parser.add_argument('--port', type=str,default="8008", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='Enable distributed training')
parser.add_argument('--mixedprec',      type=lambda x:bool(distutils.util.strtobool(x)),  default=True,  help='Enable mixed precision training')

# Logging
parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb entity')
parser.add_argument('--wandb_project', type=str, default=None, help='wandb entity')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

args = parser.parse_args()

weight_file = f'/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/weights.csv' if args.bal else None

label_metric = {'AudioSet_2M':'mAP',
             'AudioSet_20K':'mAP',
             'AudioSet_20K_Targeted':'mAP',
             'VGGSound': 'acc'}

if args.dataset in label_metric:
    args.main_metrics = label_metric[args.dataset]
else:
    args.main_metric = 'mAP'

if args.old_class_indices == "DEFAULT":
    args.old_class_indices = args.class_indices

try:
    # Read new class indices
    # We set the 'index' column as the DataFrame index for direct integer-based lookup
    class_labels_df = pd.read_csv(args.class_indices, index_col='index')
    args.label_dim = len(class_labels_df)

    # Read old class indices
    old_class_labels_df = pd.read_csv(args.old_class_indices)
    args.old_label_dim = len(old_class_labels_df)

    class_names = class_labels_df['display_name'].tolist()

    # You can choose to map 'mid' or 'display_name' to the index
    index_dict = pd.Series(class_labels_df.index.values, index=class_labels_df['mid']).to_dict()

    # Display the first few rows to confirm the structure
    print("Class labels DataFrame successfully loaded.")

    print(f"Old number of classes: {args.old_label_dim}")
    print(f"Current number of classes: {args.label_dim}")

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {args.class_indices}")
    print("Please ensure the file exists and the path is correct.")

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# The models appropriate args (some grabbed from EquiAV defaults)
args.num_mel_bins = 128 # Grabbed from ft_main audio_conf
args.target_length = 1024 # Grabbed from ft_main audio_conf
args.norm_mean = -4.346 # Grabbed from ft_main audio_conf
args.norm_std = 4.332 # Grabbed from ft_main audio_conf


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    with open(args.metadata, 'r') as f:
        data = json.load(f)

    # The JSON you provided has a 'data' key which contains a list of examples.
    examples = data.get("data", [])

    args.iteration_per_epoch = 10 # len(examples) 

    model = EquiAV_ft(**vars(args))

    if args.device == "gpu":
        model = WrappedModel(model).cuda(args.gpu)
    elif args.device == "cpu":
        model = WrappedModel(model)

    # Define the ModelTrainer
    print('\n=================Parameter of the Model=================')
    trainer = ModelTrainer(model, **vars(args))

    # Load weights, if applicable
    if args.pretrained_model is not None:
        trainer.loadParameters(args.pretrained_model)

    model.eval()  # switch to eval mode for inference

    if not examples:
        print("JSON file has no 'data' key or is empty.")
        sys.exit(1)

    for example in examples:
        audio_file_path = example.get("wav")
        labels = example.get("labels")
        if not audio_file_path:
            print(f"Skipping example with no 'wav' key: {example}")
            continue

        start_time = time.time()

        a_data = format_model_input(audio_file_path, args.target_length, args.num_mel_bins, args.norm_mean, args.norm_std)
        label_data = format_label_data(labels, index_dict, args.label_dim, args.label_smooth)

        trainer.train_on_audio_only(a_data, label_data)

        end_time = time.time()

        training_time = end_time - start_time
        print(f"Training time: {training_time}.")
    
    
    model_save_path = args.model_save_path + f"/online_model-time_{time.time()}.pth"
    trainer.saveParameters(model_save_path)
        

if __name__ == '__main__':
    main()

```

#### Continual Learning for Edge Model

The continual learning module for the edge model assumes the edge model stores a specified number of audio data examples from which it is fine-tuned to improve model performance in its local environment. The edge model must store two distributions: (1) the observed distribution of data, which is updated every time the edge model runs inference; (2) the distribution of stored data (i.e. the proportion of each class in the stored data). The following script deals with choosing whether an incoming audio file should replace one of the stored audio data examples based on the observed data distribution. Specifically, the store_audio_decision

```python
import numpy as np
import matplotlib.pyplot as plt
import random

from EquiAV.continual_learning_utils import store_audio_decision

def simulate():
    np.random.seed(0)
    random.seed(0)

    K = 5
    REPLACE_PROB = 0.7
    RESERVOIR_SIZE = 50
    STEPS = 100

    # Initial observed distribution
    obs_counts = np.random.randint(50, 200, size=K)

    # Initial stored reservoir
    stored_examples = []
    stored_counts = np.zeros(K, dtype=int)

    for _ in range(RESERVOIR_SIZE):
        num_labels = np.random.randint(1, 3)
        labels = list(np.random.choice(K, size=num_labels, replace=False))
        stored_examples.append({"labels": labels})
        for c in labels:
            stored_counts[c] += 1

    # Tracking
    stored_history = []
    obs_history = []

    # Simulate incoming data
    for step in range(STEPS):

        # Generate an incoming sample
        num_labels = np.random.randint(1, 3)
        incoming_labels = list(np.random.choice(K, size=num_labels, replace=False))

        # UPDATE OBSERVED COUNTS (important!)
        for c in incoming_labels:
            obs_counts[c] += 1

        # Decide whether to store / which to evict
        should_store, class_to_reduce, sample_to_evict = store_audio_decision(
            obs_counts,
            stored_counts,
            incoming_labels,
            stored_examples,
            REPLACE_PROB
        )

        if should_store:

            # Evict if needed
            if sample_to_evict is not None:
                old_labels = stored_examples[sample_to_evict]["labels"]
                for c in old_labels:
                    stored_counts[c] -= 1
                stored_examples[sample_to_evict] = {"labels": incoming_labels}
            else:
                stored_examples.append({"labels": incoming_labels})

            # Add new counts
            for c in incoming_labels:
                stored_counts[c] += 1

        # Track proportions
        stored_history.append(stored_counts / stored_counts.sum())
        obs_history.append(obs_counts / obs_counts.sum())

    stored_history = np.array(stored_history)
    obs_history = np.array(obs_history)

    # Plot the results
    plt.figure(figsize=(12, 7))

    # Use a fixed color map (tab10 is good for up to ~10 classes)
    colors = plt.cm.tab10(np.arange(K))

    for k in range(K):
        plt.plot(
            stored_history[:, k],
            label=f"Stored class {k}",
            color=colors[k],
            linestyle='-',
            linewidth=2
        )

        plt.plot(
            obs_history[:, k],
            label=f"Observed class {k} (true)",
            color=colors[k],
            linestyle='--',
            alpha=0.7,
            linewidth=2
        )

    plt.title("Stored vs Observed Distribution Over Time")
    plt.xlabel("Time (incoming examples)")
    plt.ylabel("Proportion")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.show()



# Run the simulation
if __name__ == "__main__":
    simulate()

```

### Dataset Preparation

To take advantage of the example scripts for offline fine-tuning and inference, please prepare the finetuning datasets. The folder structure does not matter, but create a JSON file for each dataset to use. The format of the JSON file is as follows:

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


### Offline Fine-tuning Scripts
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


### Inference Scripts

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
