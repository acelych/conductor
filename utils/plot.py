import io
import math
import torch
import colorsys
import matplotlib.pyplot as plt

from pathlib import Path
from torch import Tensor
from typing import List
from PIL import Image, ImageDraw, ImageFont

from .stat import MetricsManager

def generate_colors(n: int):
    rgbs = [colorsys.hls_to_rgb(i / n, 0.5, 0.8) for i in range(n)]
    # rgbs = [tuple(int(c * 255) for c in rgb) for rgb in rgbs]
    return rgbs

def generate_sample(colors, block_size=30, output_file='./sample.png'):
    shape = (block_size * len(colors), block_size)
    image = Image.new("RGB", shape, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    for i, color in enumerate(colors):
        loc = [
            i * block_size,
            0,
            (i + 1) * block_size,
            block_size,
        ]
        draw.rectangle(loc, fill=color)
        
    image.save(output_file)
    
def _find_close_factors(n: int):
    p, q = None, None
    for q in range(int(math.sqrt(n)), 0, -1):
        if n % q == 0:
            p = n // q
            break
    return p, q
    
class Plot:
    
    @staticmethod
    def plot_line_chart(metrics_collcet: List[MetricsManager.Metrics], save_path: str):
        get_data = lambda name: [getattr(metrics, name) for metrics in metrics_collcet]
        form = metrics_collcet[0].get_plot_form()
        chart_num = len(form.get('y_axis'))
        chart_width, chart_height = _find_close_factors(chart_num)
        fig, axs = plt.subplots(chart_height, chart_width, figsize=(chart_width * 4, chart_height * 4))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        x_axis_name: str = form.get('x_axis')
        y_axis_name: list = form.get('y_axis')
        x_axis = [getattr(metrics, x_axis_name) for metrics in metrics_collcet]
        color_num = len(y_axis_name) + len(tuple(_ for _ in y_axis_name if isinstance(_, tuple)))
        colors = generate_colors(color_num)
        locs = [(i, j) for i in range(chart_height) for j in range(chart_width)]
        for i, loc in enumerate(locs):
            yan = y_axis_name[i]
            if not isinstance(yan, tuple):
                yan = (yan,)
                color = (colors.pop(0),)
                title = yan
            else:
                color = (colors.pop(0), colors.pop(len(colors) // 2))  # opposite hue
                title = ' & '.join(yan)
            y_axises = [get_data(y) for y in yan]
            for j, y_axis in enumerate(y_axises):
                axs[loc].plot(x_axis, y_axis, color=color[j], linewidth=1, label=yan[j])
            axs[loc].set_title(title)
            axs[loc].set_xlabel(x_axis_name)
            axs[loc].set_ylabel(title)
            axs[loc].legend()
            axs[loc].grid(True)
                
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_classify_sampling(samples: List[Image.Image], label: list, pred: list, save_path: str):
        width, height = samples[0].size
        sample_num = len(samples)
        words_gap = 28
        cols, rows = _find_close_factors(sample_num)
        canvas_width, canvas_height = cols * width, rows * (height + words_gap)
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        font = ImageFont.load_default(size=10)
        draw = ImageDraw.Draw(canvas)
        
        for row in range(rows):
            for col in range(cols):
                # img
                idx = row * cols + col
                x = col * width
                y = row * (height + words_gap)
                canvas.paste(samples[idx], (x, y))
                # text
                text = f"{label[idx]}\n{pred[idx]}"
                bbox = draw.textbbox((0, 0), text=text, font=font, align="right")
                x += width - (bbox[2] - bbox[0])
                draw.text((x, y + height), text=text, font=font, align="right", fill=(0, 0, 0))
                
        canvas.save(save_path)
        
    @staticmethod
    def plot_imgs(save_path: str, imgs: List[Image.Image], label: list = None):
        width, height = imgs[0].size
        img_num = len(imgs)
        words_gap = 18 if label else 0
        cols, rows = _find_close_factors(img_num)
        canvas_width, canvas_height = cols * width, rows * (height + words_gap)
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        font = ImageFont.load_default(size=14)
        # font = ImageFont.truetype("arial.ttf", size=15)
        draw = ImageDraw.Draw(canvas)
        
        for row in range(rows):
            for col in range(cols):
                # img
                idx = row * cols + col
                x = col * width
                y = row * (height + words_gap)
                canvas.paste(imgs[idx], (x, y))
                # text
                if label:
                    text = label[idx]
                    bbox = draw.textbbox((0, 0), text=text, font=font, align="center")
                    x += (width - (bbox[2] - bbox[0])) // 2
                    # y += (words_gap - (bbox[3] - bbox[1])) // 2
                    draw.text((x, y + height), text=text, font=font, align="center", fill=(0, 0, 0))
                
        canvas.save(save_path)
        
    @staticmethod
    def plot_matrix(mat: Tensor) -> Image.Image:
        assert len(mat.shape) == 2, f"expect 2D matrix, got dimension {len(mat.shape)}"
        fig, ax = plt.subplots()
        cax = ax.imshow(mat, cmap='viridis', interpolation='nearest')
        fig.colorbar(cax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)

    @staticmethod
    def vis_matrix(mat: Tensor, minn: float = None, maxn: float = None, with_anno: bool = False) -> Image.Image:
        assert len(mat.shape) == 2, f"expect 2D matrix, got dimension {len(mat.shape)}"
        if minn is None or maxn is None:
            maxn, minn = mat.max(), mat.min()
        diff = maxn - minn
        mat_0_255 = ((mat - minn) / diff) * 255
        img = Image.fromarray(mat_0_255.to(device='cpu', dtype=torch.uint8).numpy())
        
        if with_anno:
            size = (max(img.size[0], 100), img.size[1] + 35)
            canvas = Image.new('RGB', size, (255, 255, 255))
            canvas.paste(img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            annotation = f"min: {minn:>8.4f}\nmax: {maxn:>8.4f}"
            font = ImageFont.load_default(size=14)
            bbox = draw.textbbox((0, 0), text=annotation, font=font, align="left")
            x = (size[0] - (bbox[2] - bbox[0])) // 2
            draw.text((x, img.size[1]), text=annotation, font=font, align="left", fill=(0, 0, 0))
            return canvas
        else:
            return img

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('/workspace/conductor/temp/task_1/metrics.csv')
    metrics_collcet = []
    for i, row in data.iterrows():
        metrics_collcet.append(MetricsManager.ClassifyMetrics(
            epoch = row.get('epoch'),
            time = row.get('time'),
            learn_rate = row.get('learn_rate'),
            train_loss = row.get('train_loss'),
            val_loss = row.get('val_loss'),
            top1_acc = row.get('top1_acc'),
            top5_acc = row.get('top5_acc'),
            precision = row.get('precision'),
            recall = row.get('recall'),
        ))
    Plot.plot_line_chart(metrics_collcet, './result.png')