# PyTorch reimplementation of the provided TensorFlow-based graph_view visualization utility
# Replaces Keras model handling with PyTorch nn.Module traversal
# Preserves architecture visualization in layered graph form using PIL + aggdraw

from typing import Any, Dict, List
from PIL import Image
import aggdraw
from math import ceil
import torch
import torch.nn as nn

# Replace these with your existing Box, Circle, Ellipses, ColorWheel utils
# For now we will use simplified placeholders:

class Box:
    def __init__(self):
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.fill = 'orange'
        self.outline = 'black'
    def draw(self, draw):
        brush = aggdraw.Brush(self.fill)
        pen = aggdraw.Pen(self.outline)
        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)

class Circle(Box):
    def draw(self, draw):
        brush = aggdraw.Brush(self.fill)
        pen = aggdraw.Pen(self.outline)
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)

class Ellipses(Box):
    def draw(self, draw):
        brush = aggdraw.Brush(self.fill)
        pen = aggdraw.Pen(self.outline)
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)

def graph_view_torch(model: nn.Module, to_file: str = None, node_size: int = 50, background_fill: Any = 'white',
                     padding: int = 10, layer_spacing: int = 250, node_spacing: int = 10, connector_fill: Any = 'gray',
                     connector_width: int = 1) -> Image:
    """
    Visualizes a PyTorch nn.Sequential or linear nn.Module architecture.
    """

    layers = list(model.children())
    img_width = len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding
    img_height = node_size * 5 + 2 * padding
    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)
    draw = aggdraw.Draw(img)

    current_x = padding
    for layer in layers:
        c = Box() if isinstance(layer, nn.Module) else Circle()
        c.x1 = current_x
        c.y1 = img_height // 2 - node_size // 2
        c.x2 = c.x1 + node_size
        c.y2 = c.y1 + node_size
        c.fill = 'orange'
        c.outline = 'black'
        c.draw(draw)
        current_x += node_size + layer_spacing

    draw.flush()
    if to_file:
        img.save(to_file)
    return img

# Example usage:
if __name__ == '__main__':
    # Sample model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    img = graph_view_torch(model, to_file='model_vis.png')
    img.show()