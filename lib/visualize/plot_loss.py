import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. Load the data
# Replace 'metrics.csv' with your actual filename
df = pd.read_csv('save_dir\\2026-02-17-15-17-51\\history.csv')

# 2. Clean the 'rouge' column 
# It converts "tensor(0.4851)" -> 0.4851
def clean_metric(val):
    if isinstance(val, str):
        # Use regex to find the decimal number inside the string
        match = re.search(r"(\d+\.\d+)", val)
        return float(match.group(1)) if match else 0.0
    return val

df['rouge'] = df['rouge'].apply(clean_metric)

# 3. Create the subplots
# We have 5 metrics, so a 2x3 grid works well
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
axes = axes.flatten() # Flatten to 1D for easy looping

metrics = ['loss_train', 'b1', 'b4', 'rouge', 'meteor']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']

for i, metric in enumerate(metrics):
    epochs = range(1, len(df) + 1)
    axes[i].plot(epochs, df[metric], marker='o', color=colors[i], linewidth=2)
    
    # Formatting
    axes[i].set_title(f'{metric.upper()}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Score')
    axes[i].grid(True, linestyle='--', alpha=0.7)
    axes[i].set_xticks(epochs)

# 4. Cleanup: Remove the empty 6th subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()


