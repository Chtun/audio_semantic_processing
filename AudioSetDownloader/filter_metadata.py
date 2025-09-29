import os
import json
import pandas as pd

def grab_label_ids(input_file_path: str):
    """
    Reads a label mapping CSV file, and grabs the specified label IDs from their names.

    Args:
        input_file_path (str): The path to the label mapping CSV file.
    """
    # Check if the input file exists before trying to read it.
    if not os.path.exists(input_file_path):
        print(f"Error: The file '{input_file_path}' was not found.")
        print("Please make sure the file is in the same directory as the script.")
        return

    try:
        # Read the original CSV file into a pandas DataFrame.
        # It's good practice to handle potential errors here.
        df_original = pd.read_csv(input_file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    label_ids = df_original["mid"]

    return label_ids.tolist()

def filter_label_in_metadata(json_filepath, label_ids: list):
    """
    Reads a JSON file with a 'data' key containing an array,
    modifies 'labels' list within that array, and overwrites the file.

    Args:
        json_filepath (str): The path to the JSON file (e.g., train.json, test.json).
        base_folder_in_colab (str): The base folder path in Colab
                                      (e.g., /content/drive/MyDrive/audio_data/AudioSet/train).
    """
    print(f"Processing {json_filepath}...")
    try:
        with open(json_filepath, 'r') as f:
            full_json_data = json.load(f) # Load the entire JSON object

        # Ensure the 'data' key exists and is a list
        if "data" not in full_json_data or not isinstance(full_json_data["data"], list):
            print(f"Error: '{json_filepath}' does not contain a 'data' key with a list.")
            return

        modified_entries = []
        for entry in full_json_data["data"]:
            # Your "video_id" already includes the `_start-end` part,
            # so we just need to join the base folder with it and add .wav
            if "positive_labels" in entry:
                current_labels = entry["positive_labels"].split(',')

                # Filter the list to keep only labels that are in our valid_labels set.
                filtered_labels = [label for label in current_labels if label in label_ids]
                entry["labels"] = ",".join(filtered_labels)

            modified_entries.append(entry)

        # Update the 'data' key with the modified list
        full_json_data["data"] = modified_entries

        # Write the modified data back to the same file
        with open(json_filepath, 'w') as f:
            # Use indent for pretty-printing, makes the JSON file more readable
            json.dump(full_json_data, f, indent=4)

        print(f"Successfully modified labels in {json_filepath}")
    except FileNotFoundError:
        print(f"Error: {json_filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}. Ensure it's valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_filepath}: {e}")



class_indices_path = "../EquiAV/datasets/dataprep/AudioSet_20K_Targeted/class_labels_indices-online_1.csv"
label_ids = grab_label_ids(class_indices_path)

train_metadata_path = "../EquiAV/datasets/dataprep/AudioSet_20K_Targeted/train.json"
filter_label_in_metadata(train_metadata_path, label_ids)

eval_metadata_path = "../EquiAV/datasets/dataprep/AudioSet_20K_Targeted/test.json"
filter_label_in_metadata(eval_metadata_path, label_ids)