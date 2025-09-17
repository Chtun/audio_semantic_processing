import json
import os


def change_metadata_paths(json_filepath, base_path: str):
    """
    Reads a JSON file with a 'data' key containing an array,
    modifies 'wav' paths within that array, and overwrites the file.

    Args:
        json_filepath (str): The path to the JSON file (e.g., train.json, test.json).
        base_folder_in_colab (str): The base folder path that contains the data.
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

        modified_paths = []
        for entry in full_json_data["data"]:
            # Your "video_id" already includes the `_start-end` part,
            # so we just need to join the base folder with it and add .wav
            if "video_id" in entry:
                entry["wav"] = base_path + entry["video_id"] + ".wav"

            modified_paths.append(entry)

        # Update the 'data' key with the modified list
        full_json_data["data"] = modified_paths

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


base_train_path = "./datasets/AudioSet/train/"

train_path = "../EquiAV/datasets/dataprep/AudioSet_20K_Targeted/train.json"
change_metadata_paths(train_path, base_path=base_train_path)


base_test_path = "./datasets/AudioSet/test/"
test_path = "../EquiAV/datasets/dataprep/AudioSet_20K_Targeted/test.json"
change_metadata_paths(test_path, base_path=base_test_path)