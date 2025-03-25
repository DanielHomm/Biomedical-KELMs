import matplotlib.pyplot as plt
import numpy as np

# Accuracies data for PubMedBert, BioBert, and SciBert
pubmed_accuracies = {
    "Model+Adapter EP_NB": [0.66, 0.66, 0.7, 0.52, 0.74, 0.76, 0.74, 0.6, 0.52, 0.74],
    "Model+Adapter LP_NB": [0.68, 0.58, 0.7, 0.48, 0.68, 0.7, 0.66, 0.58, 0.56, 0.72],
    "Model+Adapter LP_FUL": [0.5, 0.66, 0.72, 0.52, 0.72, 0.62, 0.58, 0.62, 0.6, 0.66],
    "Model+Adapter LP_Rel": [0.68, 0.66, 0.68, 0.5, 0.66, 0.76, 0.64, 0.62, 0.56, 0.68],
    "only model": [0.6, 0.68, 0.72, 0.52, 0.64, 0.68, 0.7, 0.68, 0.62, 0.74]
}

biobert_accuracies = {
    "Model+Adapter EP_NB": [0.6, 0.64, 0.58, 0.56, 0.58, 0.68, 0.58, 0.6, 0.6, 0.72],
    "Model+Adapter LP_NB": [0.6, 0.64, 0.62, 0.52, 0.62, 0.72, 0.58, 0.6, 0.62, 0.68],
    "Model+Adapter LP_FUL": [0.66, 0.6, 0.52, 0.42, 0.62, 0.54, 0.54, 0.5, 0.58, 0.64],
    "only model": [0.5, 0.64, 0.54, 0.5, 0.62, 0.68, 0.68, 0.54, 0.62, 0.64]
}

scibert_accuracies = {
    "Model+Adapter EP_NB": [0.56, 0.56, 0.6, 0.52, 0.58, 0.62, 0.56, 0.56, 0.68, 0.64],
    "Model+Adapter LP_NB": [0.58, 0.56, 0.6, 0.52, 0.58, 0.62, 0.56, 0.56, 0.68, 0.64],
    "Model+Adapter LP_FUL": [0.64, 0.56, 0.6, 0.52, 0.58, 0.62, 0.56, 0.56, 0.68, 0.64],
    "only model": [0.56, 0.56, 0.6, 0.52, 0.58, 0.62, 0.56, 0.56, 0.68, 0.64]
}

# Combining all datasets with appropriate labels
combined_accuracies = {
    "PubMedBert": pubmed_accuracies,
    "BioBert": biobert_accuracies,
    "SciBert": scibert_accuracies
}

# Colors for each fine-tuning method
colors = {
    "Model+Adapter EP_NB": 'lightblue',
    "Model+Adapter LP_NB": 'lightgreen',
    "Model+Adapter LP_FUL": 'lightcoral',
    "Model+Adapter LP_Rel": 'lightgoldenrodyellow',
    "only model": 'lightpink'
}

# Map colors to the box plots
labels = []
all_data = []
box_colors = []
positions = []
current_pos = 0
model_names = []

for model_name, accuracies in combined_accuracies.items():
    model_names.append(model_name)
    for key, data in accuracies.items():
        labels.append(key.replace('Adapter ', ''))
        all_data.append(data)
        box_colors.append(colors[key])
        positions.append(current_pos)
        current_pos += 0.5
    current_pos += 1  # Extra space between models

# Creating the box plot with modified spacing, colors, and smaller width, and mean line in black
fig, ax = plt.subplots(figsize=(18, 8))
box = ax.boxplot(all_data, vert=True, patch_artist=True, positions=positions, widths=0.3, showmeans=True, meanline=True)

# Setting the color for each box
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

# Setting the mean line color to black
for mean in box['means']:
    mean.set(color='black')

# Adding legend for mean and median
median_line = box['medians'][0]
mean_line = box['means'][0]
plt.legend([median_line, mean_line], ['Median', 'Mean'], loc='upper right')

# Adding model names in the middle above the boxes
model_name_positions = [(positions[len(accuracies) * i + len(accuracies) // 2]) for i in range(len(model_names))]
for model_name, model_pos in zip(model_names, model_name_positions):
    ax.text(model_pos, 1.03, model_name, ha='center', va='bottom', fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

# Adding title above the model names
plt.suptitle('Accuracies Box Plot', fontsize=16, fontweight='bold')

# Adding labels
ax.set_ylabel('Accuracy')

# Adjusting the x-ticks for better readability
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha='right')

# Displaying the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
