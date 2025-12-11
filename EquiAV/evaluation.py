from ft_trainer import *
from models.ft_EquiAV import MainModel
from datasets.AudioVisual import format_model_input
import torch
import pandas as pd
import csv
import time
import json
import os

# Step 1: Instantiate the model with appropriate args (some grabbed from EquiAV defaults)
num_mel_bins = 128 # Grabbed from ft_main audio_conf
target_length = 1024 # Grabbed from ft_main audio_conf
norm_mean = -4.346 # Grabbed from ft_main audio_conf
norm_std = 4.332 # Grabbed from ft_main audio_conf

head_type = "linear" # Head type is linear from ft model
ftmode = "audio_only" # Fine tune was done on audio only
device = 'cuda' # The device to use

# Paths to model and evaluation metadata
model_path = "./pretrained_weights/online_model/model/model_bestLoss_ft.pth"
eval_metadata_path = "./datasets/dataprep/DESED/test.json"
class_indices_path = "./datasets/dataprep/Xiao_Experiments/class_labels_indices.csv"

# Folder to save confusion matrix to.
save_folder_path = "../output/experiments"
os.makedirs(save_folder_path, exist_ok=True)

try:
    # Read the CSV file into a DataFrame
    # We set the 'index' column as the DataFrame index for direct integer-based lookup
    class_labels_df = pd.read_csv(class_indices_path, index_col='index')

    class_names = class_labels_df['display_name'].tolist()

    label_dim = len(class_labels_df)

    # Display the first few rows to confirm the structure
    print("Class labels DataFrame successfully loaded.")
    print()

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {class_indices_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()


model = MainModel(
  label_dim=label_dim,
  num_mel_bins=num_mel_bins,
  drop_out=0.0,
  drop_path=0.0,
  head_type=head_type,
  ftmode=ftmode,
)
print()

# Load state_dict from .pth file
checkpoint = torch.load(model_path, map_location=device)

# If the checkpoint is a state_dict
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
  new_key = k.replace("__M__.", "") if "__M__." in k else k
  new_state_dict[new_key] = v

# Load weights
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

# Print if anything didn't match
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)


model.eval()  # switch to eval mode for inference

with open(eval_metadata_path, 'r') as f:
  data = json.load(f)

  # The JSON you provided has a 'data' key which contains a list of examples.
  examples = data.get("data", [])
  if not examples:
    print("JSON file has no 'data' key or is empty.")
    sys.exit(1)

  confusion_matrix = {}
  inference_times = []
  predictions = []

  for class_name in class_names:
    confusion_matrix[class_name] = {}
    for other_class_name in class_names:
      confusion_matrix[class_name][other_class_name] = 0

    confusion_matrix[class_name]["Total_Positive_Examples"] = 0

  for example in examples:
    audio_file_path = example.get("wav")
    audio_id = example.get("video_id")
    if not audio_file_path:
      print(f"Skipping example with no 'wav' key: {example}")
      continue

    print(audio_file_path)

    start_time = time.time()

    a_data = format_model_input(audio_file_path, target_length, num_mel_bins, norm_mean, norm_std)
    v_data = torch.zeros([3, 1, 1]) + 0.01

    outputs = model.forward(a_data, v_data)

    confidence_scores = torch.sigmoid(outputs)
    predicted_indices = (confidence_scores > 0.5).nonzero(as_tuple=False)[:,1]
    print(predicted_indices)
    predicted_labels = class_labels_df.loc[predicted_indices.tolist()]['display_name'].tolist()
    # Get the indices of the top 3 scores
    print(confidence_scores.squeeze())
    top3_scores, top3_indices = torch.topk(confidence_scores.squeeze(), k=3)
    top3_scores = top3_scores.detach().cpu().tolist()

    # Map indices to label names
    top3_labels = class_labels_df.loc[top3_indices.tolist()]['display_name'].tolist()

    ground_truth_labels_str = example.get("labels")
    ground_truth_mid_labels = ground_truth_labels_str.split(',')
    mid_to_display_name_df = class_labels_df.set_index('mid')
    ground_truth_label_names = mid_to_display_name_df.loc[ground_truth_mid_labels]['display_name'].tolist()

    end_time = time.time()

    inference_time = end_time - start_time
    inference_times.append(inference_time)

    predictions.append((audio_id, ground_truth_label_names, top3_labels, top3_scores))

    print("Predictions: ")
    print(predicted_labels)

    print()

    print("Top 3 labels and scores:")
    print(top3_labels)
    print(top3_indices)
    print(top3_scores)

    print()

    print("Ground Truth: ")
    print(ground_truth_label_names)

    for predicted_label in predicted_labels:
      if predicted_label in ground_truth_label_names:
        confusion_matrix[predicted_label][predicted_label] += 1
      else:
        for ground_truth_label in ground_truth_label_names:
          confusion_matrix[ground_truth_label][predicted_label] += 1

    for ground_truth_label in ground_truth_label_names:
      confusion_matrix[ground_truth_label]["Total_Positive_Examples"] += 1

    print()

  confusion_matrix_path = os.path.join(save_folder_path, "confusion_matrix.csv")
  with open(confusion_matrix_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    all_column_headers = class_names + ["Total_Positive_Examples"]
    writer.writerow(['Actual \\ Predicted'] + all_column_headers)
    for class_name in class_names:
      row = [class_name] + [confusion_matrix[class_name].get(pred, 0) for pred in all_column_headers]
      writer.writerow(row)

  inference_times_path = os.path.join(save_folder_path, "inference_times.csv")
  with open(inference_times_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Inference Time (seconds)"])
    for time in inference_times:
      writer.writerow([time])

  # Path to save CSV
  predictions_csv_path = os.path.join(save_folder_path, "predictions.csv")
  with open(predictions_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile)
      # Header
      writer.writerow([
          "Audio File",
          "Ground Truth Labels",
          "Top 3 Labels", 
          "Top 3 Confidence"
      ])
      
      for audio_id, ground_truth_labels, top3_labels, top3_scores in predictions:
          # Convert lists to comma-separated strings
          top3_labels_str = ", ".join(top3_labels)
          # Convert tensor of scores to list of floats with 3 decimal precision
          top3_scores_str = ", ".join([f"{score:.3f}" for score in top3_scores])
          ground_truth_str = ", ".join(ground_truth_labels)
          
          writer.writerow([
              audio_id,
              ground_truth_str,
              top3_labels_str, 
              top3_scores_str
          ])


  average_inference_time = sum(inference_times) / len(inference_times)
  print(f"Average inference time: {average_inference_time} seconds")

