import matplotlib.pyplot as plt
import torch

def plot_raw(train_data):
    """
    Visualizes a random selection of images and their associated captions from the dataset.

    Args:
        train_data (Dataset): The dataset instance to sample from, returning (image, label) pairs.
    """
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 1, 3
    count = 1
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        img = img.pixel_values[0]
        print(img)
        ax = figure.add_subplot(rows, 2, count)
        ax.imshow(img.permute(1, 2, 0))
        count += 1
        ax = figure.add_subplot(rows,2,count)

        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,len(label))
        for i, caption in enumerate(label):
            ax.text(0,i,caption,fontsize=10)
        count += 1
    plt.show()