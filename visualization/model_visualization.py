from ft_trainer import *
from models.ft_EquiAV import MainModel
from datasets.AudioVisual import format_model_input
import torch
import pandas as pd
import csv
import time
import json
import os
from torch.utils.tensorboard import SummaryWriter

# The models appropriate args (some grabbed from EquiAV defaults)
label_dim = 21 # 21 target classes for AudioSet_20K_Targeted
num_mel_bins = 128 # Grabbed from ft_main audio_conf
target_length = 1024 # Grabbed from ft_main audio_conf
norm_mean = -4.346 # Grabbed from ft_main audio_conf
norm_std = 4.332 # Grabbed from ft_main audio_conf

device = "cpu"
head_type = "linear" # Head type is linear from ft model
ftmode = "audio_only" # Fine tune was done on audio only

model = MainModel(
  label_dim=label_dim,
  num_mel_bins=num_mel_bins,
  drop_out=0.0,
  drop_path=0.0,
  head_type=head_type,
  device=device,
  ftmode=ftmode
)

try:
    # Read the CSV file into a DataFrame
    # We set the 'index' column as the DataFrame index for direct integer-based lookup
    class_labels_df = pd.read_csv(class_indices_path, index_col='index')

    class_names = class_labels_df['display_name'].tolist()

    # Display the first few rows to confirm the structure
    print("Class labels DataFrame successfully loaded.")

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {class_indices_path}")
    print("Please ensure the file exists and the path is correct.")


model.eval()  # switch to eval mode for inference

a_data = torch.randn(1, 3, num_mel_bins, target_length)

v_data = torch.zeros([3, 1, 1]) + 0.01

a_data = a_data.to(device)
v_data = v_data.to(device)

writer = SummaryWriter('runs/model_graph_visualization')

# The code from your snippet
with torch.no_grad():
    writer.add_graph(model, (a_data, v_data))

# Close the writer
writer.close()

    