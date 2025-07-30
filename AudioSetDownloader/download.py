from Downloader import *

download_path = "..\\EquiAV\\datasets\\AudioSet"

d = Downloader(
  root_path=download_path,
  labels=[
          "Speech",
          "Male speech, man speaking",
          "Female speech, woman speaking",
          "Walk, footsteps",
          "Breathing",
          "Cough",
          "Dog",
          "Cat",
          "Bark",
          "Meow",
          "Inside, small room",
          "Water",
          "Water tap, faucet",
          "Toilet flush",
          "Sink (filling or washing)",
          "Music",
          "Vacuum cleaner",
          "Microwave oven",
          "Dishes, pots, and pans",
          "Door",
          "Television"
          ],
  n_jobs=40,
  download_type='eval',
  copy_and_replicate="False",
)

d.download(
  save_folder_name="train",
  format='wav',
  max_labels_per_class=2,
  seed=10
)

d.download(
  save_folder_name="test",
  format='wav',
  max_labels_per_class=2,
  seed=100
)