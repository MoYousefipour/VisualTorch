from typing import Callable, Any
import aggdraw
from PIL import Image, ImageDraw, ImageFont
from math import ceil
import torch
import torch.nn as nn
import warnings

from .utils import *
from .layer_utils import *


class Box(RectShape):
    de: int
    shade: int

    def draw(self, draw: ImageDraw, draw_reversed: bool = False):
        pen, brush = self._get_pen_brush()

        # Cast coordinates to int for aggdraw compatibility
        x1, y1, x2, y2 = map(int, (self.x1, self.y1, self.x2, self.y2))
        de = int(self.de)

        if hasattr(self, 'de') and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

            if draw_reversed:
                draw.line([x2 - de, y1 - de, x2 - de, y2 - de], pen)
                draw.line([x2 - de, y2 - de, x2, y2], pen)
                draw.line([x1 - de, y2 - de, x2 - de, y2 - de], pen)

                draw.polygon([x1, y1,
                              x1 - de, y1 - de,
                              x2 - de, y1 - de,
                              x2, y1
                              ], pen, brush_s1)

                draw.polygon([x1 - de, y1 - de,
                              x1, y1,
                              x1, y2,
                              x1 - de, y2 - de
                              ], pen, brush_s2)
            else:
                draw.line([x1 + de, y1 - de, x1 + de, y2 - de], pen)
                draw.line([x1 + de, y2 - de, x1, y2], pen)
                draw.line([x1 + de, y2 - de, x2 + de, y2 - de], pen)

                draw.polygon([x1, y1,
                              x1 + de, y1 - de,
                              x2 + de, y1 - de,
                              x2, y1
                              ], pen, brush_s1)

                draw.polygon([x2 + de, y1 - de,
                              x2, y1,
                              x2, y2,
                              x2 + de, y2 - de
                              ], pen, brush_s2)

        draw.rectangle([x1, y1, x2, y2], pen, brush)


def layered_view(model: nn.Module,
                 to_file: str = None,
                 min_z: int = 20,
                 min_xy: int = 20,
                 max_z: int = 400,
                 max_xy: int = 2000,
                 scale_z: float = 1.5,
                 scale_xy: float = 4,
                 type_ignore: list = None,
                 index_ignore: list = None,
                 color_map: dict = None,
                 one_dim_orientation: str = 'z',
                 index_2D: list = [],
                 background_fill: Any = 'white',
                 draw_volume: bool = True,
                 draw_reversed: bool = False,
                 padding: int = 20,  # Increased padding for safety
                 text_callable: Callable[[int, nn.Module], tuple] = None,
                 text_vspacing: int = 4,
                 spacing: int = 10,
                 draw_funnel: bool = True,
                 shade_step=10,
                 legend: bool = False,
                 legend_text_spacing_offset=15,
                 font: ImageFont = None,
                 font_color: Any = 'black',
                 show_dimension=False,
                 sizing_mode: str = 'accurate',
                 dimension_caps: dict = None,
                 relative_base_size: int = 20) -> Image:
    """
    Generates a layered architecture visualization for a given PyTorch model.
    """

    # Collect output shapes via forward hooks
    shapes = []

    def hook(module, inp, out):
        shapes.append(out.shape)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d, nn.Flatten)):
            hooks.append(layer.register_forward_hook(hook))

    # Run dummy forward pass to activate hooks and collect shapes
    try:
        # Create a dummy input tensor with batch size 1 matching model input
        first_layer = next(model.modules())
        if hasattr(model, 'forward'):
            input_shape = None
            for m in model.modules():
                if hasattr(m, 'in_features'):
                    input_shape = (1, m.in_features)
                    break
                elif hasattr(m, 'in_channels'):
                    # default spatial size
                    input_shape = (1, m.in_channels, 64, 64)
                    break
            if input_shape is None:
                input_shape = (1, 3, 64, 64)
            dummy_input = torch.randn(input_shape)
            model(dummy_input)
    except Exception as e:
        warnings.warn(f"Dummy forward pass failed: {e}")

    # Remove hooks to avoid side effects
    for h in hooks:
        h.remove()

    boxes = []
    layer_y = []
    color_wheel = ColorWheel()
    current_z = padding
    x_off = -1

    layer_types = []
    dimension_list = []
    img_height = 0
    max_right = 0

    if type_ignore is None:
        type_ignore = []
    if index_ignore is None:
        index_ignore = []
    if color_map is None:
        color_map = {}

    layers_list = [layer for layer in model.modules() if layer != model]

    for idx, (layer, shape) in enumerate(zip(layers_list, shapes)):

        if type(layer) in type_ignore or idx in index_ignore:
            continue

        layer_type = type(layer)

        if legend and show_dimension:
            layer_types.append(layer_type)
        elif layer_type not in layer_types:
            layer_types.append(layer_type)

        x, y, z = calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
                                             one_dim_orientation, sizing_mode, dimension_caps, relative_base_size)

        box = Box()
        box.de = 0
        if draw_volume and idx not in index_2D:
            box.de = x / 3
        if x_off == -1:
            x_off = box.de / 2

        box.x1 = current_z - box.de / 2
        box.y1 = box.de
        box.x2 = box.x1 + z
        box.y2 = box.y1 + y

        # Clamp box coords to stay within image bounds (will clamp after img creation)
        # But for now just keep track for image sizing
        box.fill = color_map.get(layer_type, {}).get(
            'fill', color_wheel.get_color(layer_type))
        box.outline = color_map.get(layer_type, {}).get('outline', 'black')
        color_map[layer_type] = {'fill': box.fill, 'outline': box.outline}
        box.shade = shade_step

        boxes.append(box)
        layer_y.append(box.y2 - (box.y1 - box.de))
        hh = box.y2 - (box.y1 - box.de)
        img_height = max(img_height, hh)
        max_right = max(max_right, box.x2 + box.de)
        current_z += z + spacing

    img_width = max_right + x_off + padding
    img_height = max(img_height, hh)
    img = Image.new('RGBA', (int(ceil(img_width)),
                    int(ceil(img_height))), background_fill)

    # Adjust boxes positions and clamp inside image boundary
    for i, node in enumerate(boxes):
        y_off = (img.height - layer_y[i]) / 2
        node.y1 += y_off
        node.y2 += y_off
        node.x1 += x_off
        node.x2 += x_off

        node.x1 = max(0, min(node.x1, img.width - 1))
        node.x2 = max(0, min(node.x2, img.width - 1))
        node.y1 = max(0, min(node.y1, img.height - 1))
        node.y2 = max(0, min(node.y2, img.height - 1))

    draw = aggdraw.Draw(img)
    last_box = None

    for box in boxes:
        pen = aggdraw.Pen(get_rgba_tuple(box.outline))
        if last_box is not None and draw_funnel:
            draw.line([last_box.x2 + last_box.de, last_box.y1 - last_box.de,
                       box.x1 + box.de, box.y1 - box.de], pen)
            draw.line([last_box.x2 + last_box.de, last_box.y2 - last_box.de,
                       box.x1 + box.de, box.y2 - box.de], pen)
            draw.line([last_box.x2, last_box.y2, box.x1, box.y2], pen)
            draw.line([last_box.x2, last_box.y1, box.x1, box.y1], pen)

        box.draw(draw, draw_reversed=False)
        last_box = box

    draw.flush()

    if to_file is not None:
        img.save(to_file)

    return img
