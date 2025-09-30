import os
import sys
import time
import json
import torch
import pandas as pd
from ft_trainer import *
from models.ft_EquiAV import MainModel
from datasets.AudioVisual import format_model_input
import csv
import argparse


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


def load_audio_model(
        class_indices_path: str,
        weights_path: str,
        device: str="cpu",
        head_type: str="linear",
        modality_mode: str="audio_only",
        num_mel_bins: int=128,
    ) -> tuple[MainModel, pd.DataFrame]:
    """
    Returns an audio model loaded with the specified fine-tuned weights.

    Args:
        class_indices_path: The path to the class (label) string to index pairing.
        weights_path: The path to the fine-tuned weights for EquiAV model.
        device: The device to run the model on ('cpu' or 'gpu').
        head_type: The type of classification head to use ('linear' or 'mlp').
        modality_mode: Which type of input modalities to use ('audio_only', 'video_only', or 'multimodal' for video and audio)
        num_mel_bins: The number of bins for the mel spectrogram.

    Returns:
        1. The EquiAV model with the specified fine-tuned weights loaded.
        2. The DataFrame for the class indices pairing.
    """

    # ----------------------------
    # Load class labels
    # ----------------------------
    try:
        class_labels_df = pd.read_csv(class_indices_path, index_col='index')
        label_dim = len(class_labels_df)
        print("Class labels loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {class_indices_path}")
        sys.exit(1)

    # ----------------------------
    # Load model
    # ----------------------------
    model = MainModel(
        label_dim=label_dim,
        num_mel_bins=num_mel_bins,
        drop_out=0.0,
        drop_path=0.0,
        head_type=head_type,
        ftmode=modality_mode
    )

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("__M__.", "") if "__M__." in k else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)

    return model, class_labels_df


def audio_event_classification(
        wav_files: dict[str, str],
        model: MainModel,
        class_labels_df: pd.DataFrame,
        target_length: int=1024,
        num_mel_bins: int=128,
        norm_mean: float=-4.346,
        norm_std: float=4.332,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Performs multi-class classification for a set of wav files using EquiAV model.

    Args:
        wav_files: A dictionary of ID to wav file path pairings.
        model: The EquiAV model.
        class_labels_df: The dataframe of the class (label) name to index pairings.
        target_length: The maximum length of the audio.
        num_mel_bins: The number of bins for the mel spectrogram.
        norm_mean: The mean of the audio values for normalization.
        norm_std: The standard deviation of the audio values for normalization.

    Returns:
        1. A dictionary of ID to predicted classes for each wav file.
        2. A dictionary of ID to inference time for each wav file.
    """

    model.eval()

    results = {}
    inference_times = {}

    current_example_idx = 0

    for id in wav_files.keys():
        audio_file_path = wav_files[id]
        if not audio_file_path:
            continue

        start_time = time.time()

        a_data = format_model_input(audio_file_path, target_length, num_mel_bins, norm_mean, norm_std)
        v_data = torch.zeros([3, 1, 1]) + 0.01  # Dummy visual input

        with torch.no_grad():
            outputs = model.forward(a_data, v_data)
            confidence_scores = torch.sigmoid(outputs)
            predicted_indices = (confidence_scores > 0.5).nonzero(as_tuple=False)[:, 1]
            predicted_labels = class_labels_df.loc[predicted_indices.tolist()]['display_name'].tolist()

        end_time = time.time()
        inference_time = end_time - start_time
        inference_times[id] = inference_time

        current_example_idx += 1
        print(f"Finished {current_example_idx}/{len(wav_files)} examples.")
        
        results[id] = predicted_labels

    return results, inference_times



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