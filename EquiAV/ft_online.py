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
parser.add_argument('--class_indices',type=str, default="./datasets/dataprep/AudioSet_20K_Targeted/class_labels_indices.csv", help='Path of dataset class index mapping.')
parser.add_argument('--old_class_indices', type=str, default='./datasets/dataprep/AudioSet_20K_Targeted/class_labels_indices.csv')
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

args.main_metrics = label_metric[args.dataset]

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
        v_data = torch.zeros([3, 1, 1]) + 0.01
        label_data = format_label_data(labels, index_dict, args.label_dim, args.label_smooth)

        trainer.train_on_single_pair(a_data, v_data, label_data)

        end_time = time.time()

        training_time = end_time - start_time
        print(f"Training time: {training_time}.")
    
    
    model_save_path = args.model_save_path + f"/online_model-time_{time.time()}.pth"
    trainer.saveParameters(model_save_path)
        

if __name__ == '__main__':
    main()