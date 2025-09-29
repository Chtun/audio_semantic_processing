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

# ----------------------------
# Config
# ----------------------------
num_mel_bins = 128
target_length = 1024
norm_mean = -4.346
norm_std = 4.332

head_type = "linear"
ftmode = "audio_only"
device = 'cpu'

model_path = "./pretrained_weights/online_model/model/model_bestLoss_ft.pth"
eval_metadata_path = "./datasets/dataprep/Xiao_Experiments/test.json"
class_indices_path = "./datasets/dataprep/Xiao_Experiments/class_labels_indices.csv"

output_csv_path = "../output/experiments/predictions.csv"

# ----------------------------
# Load class labels
# ----------------------------
try:
    class_labels_df = pd.read_csv(class_indices_path, index_col='index')
    class_names = class_labels_df['display_name'].tolist()
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
    ftmode=ftmode
)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("__M__.", "") if "__M__." in k else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# ----------------------------
# Load evaluation metadata
# ----------------------------
with open(eval_metadata_path, 'r') as f:
    data = json.load(f)
    examples = data.get("data", [])
    if not examples:
        print("No examples found in JSON.")
        sys.exit(1)

# ----------------------------
# Run inference
# ----------------------------
results = []
inference_times = {}

current_example_idx = 0

for example in examples:
    video_id = example.get("video_id") or os.path.splitext(os.path.basename(example.get("wav", "")))[0]
    audio_file_path = example.get("wav")
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
    inference_times[video_id] = inference_time

    current_example_idx += 1
    print(f"Finished {current_example_idx}/{len(examples)} examples.")
    

    predicted_label_str = ",".join(predicted_labels)
    results.append({"video_id": video_id, "predicted_label": predicted_label_str})

# ----------------------------
# Save predictions to CSV
# ----------------------------
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["video_id", "predicted_label"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved predictions for {len(results)} examples to {output_csv_path}")

# ----------------------------
# Print/save inference times
# ----------------------------
average_time = sum(inference_times.values()) / len(inference_times)
print(f"Average inference time per example: {average_time:.4f} seconds")
