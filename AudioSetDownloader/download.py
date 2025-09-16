from Downloader import *

download_path = "..\\EquiAV\\datasets\\AudioSet"

d_train = Downloader(
  root_path=download_path,
  labels=[
          # "Speech",
          # "Male speech, man speaking",
          # "Female speech, woman speaking",
          # "Walk, footsteps",
          # "Breathing",
          # "Cough",
          # "Dog",
          # "Cat",
          # "Bark",
          # "Meow",
          # "Inside, small room",
          # "Water",
          # "Water tap, faucet",
          # "Toilet flush",
          # "Sink (filling or washing)",
          # "Music",
          # "Vacuum cleaner",
          # "Microwave oven",
          # "Dishes, pots, and pans",
          # "Door",
          # "Television",
          "Doorbell"
          ],
  n_jobs=16,
  download_type='unbalanced_train',
  copy_and_replicate="False",
)

d_train.download(
  save_folder_name="train",
  format='wav',
  max_labels_per_class=20,
  seed=10
)

d_test = Downloader(
  root_path=download_path,
  labels=[
          # "Speech",
          # "Male speech, man speaking",
          # "Female speech, woman speaking",
          # "Walk, footsteps",
          # "Breathing",
          # "Cough",
          # "Dog",
          # "Cat",
          # "Bark",
          # "Meow",
          # "Inside, small room",
          # "Water",
          # "Water tap, faucet",
          # "Toilet flush",
          # "Sink (filling or washing)",
          # "Music",
          # "Vacuum cleaner",
          # "Microwave oven",
          # "Dishes, pots, and pans",
          # "Door",
          # "Television",
          "Doorbell"
          ],
  n_jobs=16,
  download_type='eval',
  copy_and_replicate="False",
)

d_test.download(
  save_folder_name="test",
  format='wav',
  max_labels_per_class=10,
  seed=100
)