import matplotlib.pyplot as plt
import numpy as np

# Data
times = {
    "PubMedBert": [1167, 1167, 372, 1168, 294],
    "BioBert": [1179, 1180, 474, 299],  # Removed LP_Rel
    "SciBert": [768, 294, 294, 298]     # Removed LP_Rel
}

labels_pubmedbert = ["Model+EP_NB", "Model+LP_NB", "Model+LP_FUL", "Model+LP_Rel", "only model"]
labels_other = ["Model+EP_NB", "Model+LP_NB", "Model+LP_FUL", "only model"]

# Define colors for each fine-tuning approach
colors = {
    "Model+EP_NB": "lightblue",
    "Model+LP_NB": "lightgreen",
    "Model+LP_FUL": "lightcoral",
    "Model+LP_Rel": "lightgoldenrodyellow",
    "only model": "lightpink"
}

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Generate bars
bar_width = 0.2
space_within_model = 0.05
space_between_models = 0.5

all_positions = []
all_labels = []

for i, (model, time) in enumerate(times.items()):
    if model == "PubMedBert":
        positions = np.arange(len(time)) * (bar_width + space_within_model) + i * (len(labels_pubmedbert) * (bar_width + space_within_model) + space_between_models)
        current_labels = labels_pubmedbert
    else:
        positions = np.arange(len(time)) * (bar_width + space_within_model) + i * (len(labels_other) * (bar_width + space_within_model) + space_between_models)
        current_labels = labels_other

    for j, t in enumerate(time):
        if not np.isnan(t):
            ax.bar(positions[j], t, bar_width, color=colors[current_labels[j]])
    all_positions.extend(positions)

    # Add model name as annotation once above the middle of the group
    model_position = (positions[0] + positions[-1]) / 2
    max_time = max([t for t in time if not np.isnan(t)])
    ax.text(model_position, max_time + 100, model, ha='center', va='bottom', fontsize=12, fontweight='bold')

# Labeling
ax.set_ylabel('Time in Seconds')
ax.set_title('Mean Times for Fine Tuning', pad=50)  # Add space between title and plot

# Setting positions and labels for xticks
flat_labels = []
for model in times.keys():
    if model == "PubMedBert":
        flat_labels.extend(labels_pubmedbert)
    else:
        flat_labels.extend(labels_other)

flat_positions = [position for sublist in [np.arange(len(time)) * (bar_width + space_within_model) + i * (len(labels_pubmedbert) * (bar_width + space_within_model) + space_between_models) if list(times.keys())[i] == "PubMedBert" else np.arange(len(time)) * (bar_width + space_within_model) + i * (len(labels_other) * (bar_width + space_within_model) + space_between_models) for i, time in enumerate(times.values())] for position in sublist]

ax.set_xticks(flat_positions)
ax.set_xticklabels(flat_labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()
