import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_line_chart(epochs: np.ndarray, train_loss: np.ndarray, val_loss: np.ndarray, precision: np.ndarray, recall: np.ndarray, top1: np.ndarray):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    axs[0, 0].plot(epochs, train_loss, color='blue', linewidth=1, label='Train Loss')
    axs[0, 0].plot(epochs, val_loss, color='red', linewidth=1, label='Val Loss')
    axs[0, 0].set_title('Train Loss vs Val Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, precision, color='green', linewidth=1, label='Precision')
    axs[0, 1].set_title('Precision')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, recall, color='purple', linewidth=1, label='Recall')
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, top1, color='orange', linewidth=1, label='Top-1')
    axs[1, 1].set_title('Top-1 Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Top-1')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.savefig('./temp0.png')
    plt.close()

df: pd.DataFrame = pd.read_csv(r'C:\Workspace\DataScience\ComputerVision\conductor\temp\task_handmake_cosine_64_enhanced_better\metrics.csv')
plot_line_chart(
    np.array(df.get('epoch')),
    np.array(df.get('train_loss')),
    np.array(df.get('val_loss')),
    np.array(df.get('precision')),
    np.array(df.get('recall')),
    np.array(df.get('top1_acc'))
    )