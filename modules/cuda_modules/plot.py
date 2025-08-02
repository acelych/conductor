import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# # 模拟数据
# B_values = range(1, 65)  # 批量大小B
# C_values = range(1, 65)  # 通道数C
# HW_fixed = 256

# np.random.seed(42)
# algo1_speed = np.random.uniform(50, 200, size=(len(C_values), len(B_values)))
# algo2_speed = np.random.uniform(50, 200, size=(len(C_values), len(B_values)))
# speed_ratio = algo1_speed / algo2_speed

colors = [
    (0.0, "#0B85FF"),
    (0.5, "#E4E4E4"),
    (1.0, "#FF4E4E")
]

# normalized_colors = [(x/5.0, color) for x, color in colors]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)

def draw_heatmap(data, size, name):
    fig, ax = plt.subplots(figsize=(size, size))

    heatmap = sns.heatmap(
        data,
        cmap=cmap,
        square=True,
        cbar=True,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    plt.savefig(f"{name}.svg", format="svg", bbox_inches="tight")

data = np.load("/workspace/conductor/modules/cuda_modules/exp_res.npz")
norm_diff = lambda mat: (mat[0]-mat[1]) / (mat[0]+mat[1]+1e-9)
draw_heatmap(norm_diff(data['bc']), data['bc'].shape[1], "batch_channel")
draw_heatmap(norm_diff(data['be']), data['be'].shape[1], "batch_edge")
draw_heatmap(norm_diff(data['ce']), data['ce'].shape[1], "channel_edge")