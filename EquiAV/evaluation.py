from models.ft_EquiAV import MainModel
from datasets.AudioVisual import format_model_input
import torch
import os

# These are derived from the AudioSet 2M configurations as shown in ft_main
norm_stats = [-4.346, 4.332] # [mean, std]
target_length = 1024
label_dim = 527
num_mel_bins = 128
drop_out = 0
drop_path = 0.1

print(label_dim)


# Instantiate the model with appropriate args
model = MainModel(label_dim=label_dim, num_mel_bins=num_mel_bins, drop_out=drop_out, drop_path=drop_path)

# Load state_dict from .pth file
model_checkpoint_path = "pretrained_weights\EquiAV_pretrained.pth"
checkpoint = torch.load(model_checkpoint_path, map_location="cpu")  # or "cuda"

model.ftmode = "audio_only"


state_dict = checkpoint

# Format keys
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("__M__.", "") if k.startswith("__M__.") else k
    new_state_dict[new_key] = v

print()
print()

# Load weights
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)



model.eval()

audio_set_data_path = "datasets/AudioSet/"

if not os.path.exists(audio_set_data_path):
    print(f"Error: Base path '{audio_set_data_path}' does not exist.")
    print("Please ensure your 'datasets/AudioSet/' directory is correctly set up.")
else:
    for item_name in os.listdir(audio_set_data_path):
        item_path = os.path.join(audio_set_data_path, item_name)

        if os.path.isdir(item_path):
            item_label = item_name  # The name of the folder is the label of the item
            print(f"\n--- Processing folder with labels for: {item_label} ---")

            for file_name in os.listdir(item_path):
                if file_name.endswith(".wav"):
                    input_audio_file_path = os.path.join(item_path, file_name)
                    print(f"  Found WAV file: {input_audio_file_path}")

                    # generate formatted fbanks from wav file.
                    audio_data = format_model_input(
                        file_path=input_audio_file_path,
                        target_length=target_length,
                        num_mel_bins=num_mel_bins,
                        norm_mean=norm_stats[0],
                        norm_std=norm_stats[1]
                    )

                    video_data = None

                    # Run forward pass on the fbanks
                    prediction = model.forward(a=audio_data, v=video_data)

                    print(f"Prediction shape: {prediction.shape}")

                    # print(f"Prediction for the audio file: {prediction}")

                else:
                    print(f"  Skipping non-WAV file: {file_name}")
        else:
            print(f"--- Skipping non-directory item: {item_name} ---")

print("\nFinished processing all folders and WAV files.")