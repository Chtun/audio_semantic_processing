# AudioSet Downloader

This folder contains scripts for downloading the AudioSet dataset, or subsets of it.

## Getting Started

First, run download.py. In download.py, specify the path to download to ("download_path"), the number of examples to grab per class, and the set of classes you want to download.

Then, use change_metadata_paths.py to change the .wav paths in the metadata for the dataset. Specify the base training and test folders that contains the .wav files ("base_train_path" and "base_test_path"), as well as the paths to the metadata ("train_path" and "test_path").

Finally, use filter_metadata.py to filter the labels in each download to only include the labels that are part of your targeted classes (if you are not using the full 527 classes, but only a subset).
