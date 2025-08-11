import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Load CSV
output_dir = "/home/chtun/Documents/AudioSemanticModel/audio_semantic_processing/output/"
input_dir = "/home/chtun/Documents/AudioSemanticModel/audio_semantic_processing/output/"

confusion_matrix_path = os.path.join(input_dir, "confusion_matrix.csv")
confusion_matrix_image_path = os.path.join(output_dir, "confusion_matrix.jpg")

inference_times_path = os.path.join(input_dir, "inference_times.csv")
inference_times_image_path = os.path.join(output_dir, "inference_times.jpg")

df = pd.read_csv(confusion_matrix_path, index_col=0)

df_normalized = df.copy()

if "Total_Positive_Examples" not in df.columns:
    print("Error: 'Total_Positive_Examples' column not found in the DataFrame.")
else:
    columns_to_normalize = df.columns.drop("Total_Positive_Examples")

    df_normalized.loc[:, columns_to_normalize] = df.loc[:, columns_to_normalize].div(
        df["Total_Positive_Examples"], axis=0
    ).replace(np.inf, np.nan)

df_normalized.drop("Total_Positive_Examples", axis=1, inplace=True)

# Set plot size
plt.figure(figsize=(22, 18))

# Create heatmap
sns.heatmap(df_normalized, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5,
            cbar_kws={'label': 'Count'}, linecolor='gray')

# Axis labels and title
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Heatmap")

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()



plt.savefig(confusion_matrix_image_path)

# Read inference times from CSV
inference_times = []
with open(inference_times_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        if row:  # skip empty rows
            inference_times.append(float(row[0]))

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(inference_times, bins=20, edgecolor='black', alpha=0.7)

plt.xlabel("Inference Time (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Inference Times")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(inference_times_image_path)