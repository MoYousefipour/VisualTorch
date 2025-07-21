from typing import Callable
from PIL import Image, ImageDraw,ImageFont
from math import ceil
import torch
import torch.nn as nn


from .utils import *
from .layer_utils import *


def layered_view(model: nn.Module,
                 input_shape: tuple = None,
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
                 index_2d=None,
                 background_fill: Any = 'white',
                 draw_volume: bool = True,
                 draw_reversed: bool = False,
                 padding: int = 20,
                 text_callable: Callable[[int, nn.Module], tuple] = None,
                 text_vspacing: int = 4,
                 spacing: int = 10,
                 draw_funnel: bool = True,
                 shade_step=10,
                 legend: bool = False,
                 font: ImageFont = None,
                 font_color: Any = 'black',
                 show_dimension=False,
                 sizing_mode: str = 'accurate',
                 dimension_caps: dict = None,
                 relative_base_size: int = 20) -> Image:
    """
    Generates a layered architecture visualization for a given PyTorch model.
    Args:
        model (nn.Module): The PyTorch model to visualize.
        input_shape (tuple): Shape of the input tensor to the model.
        to_file (str): If provided, saves the image to this file path.
        min_z (int): Minimum width of the layer boxes.
        min_xy (int): Minimum height of the layer boxes.
        max_z (int): Maximum width of the layer boxes.
        max_xy (int): Maximum height of the layer boxes.
        scale_z (float): Scaling factor for the width of the layer boxes.
        scale_xy (float): Scaling factor for the height of the layer boxes.
        type_ignore (list): List of layer types to ignore in visualization.
        index_ignore (list): List of layer indices to ignore in visualization.
        color_map (dict): Custom color mapping for layer types.
        one_dim_orientation (str): Orientation for one-dimensional layers ('x', 'y', 'z').
        index_2d (list): List of layer indices that should be treated as 2D layers.
        background_fill (Any): Background color for the image.
        draw_volume (bool): Whether to draw volume for 3D layers.
        draw_reversed (bool): Whether to draw boxes in reverse order.
        padding (int): Padding around the image.
        text_callable (Callable): Function to generate text for each layer.
        text_vspacing (int): Vertical spacing for layer text.
        spacing (int): Spacing between layer boxes.
        draw_funnel (bool): Whether to draw funnel lines between layers.
        shade_step (int): Step size for shading the boxes.
        legend (bool): Whether to include a legend for layer types.
        font (ImageFont): Font to use for layer text.
        font_color (Any): Color of the font for layer text.
        show_dimension (bool): Whether to show dimensions in the layer boxes.
        sizing_mode (str): Sizing mode for the layer boxes ('accurate', 'balanced', etc.).
        dimension_caps (dict): Caps for dimensions to prevent excessive sizes.
        relative_base_size (int): Base size for relative dimension scaling.
    Returns:
        Image: The generated image of the model architecture.
    """

    # Collect output shapes via forward hooks
    if index_2d is None:
        index_2d = []

    hh,counter=0,0
    shapes = []

    def hook(_, __, out):
        shapes.append(out.shape)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d, nn.Flatten)):
            hooks.append(layer.register_forward_hook(hook))

    # Run dummy forward pass to activate hooks and collect shapes
    try:
        # Create a dummy input tensor with batch size 1 matching model input
        next(model.modules())
        if hasattr(model, 'forward'):
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
    
    compiled_list=[]
    # Start Building Tikz
    for idx , (layer,shape) in enumerate(zip(layers_list,shapes)):
        if type(layer) in type_ignore or idx in index_ignore:
            continue
        x, y, z = calculate_layer_dimensions(
            shape, scale_z, scale_xy,
            max_z, max_xy, min_z, min_xy,
            one_dim_orientation, sizing_mode,
            dimension_caps, relative_base_size
        )
        
        layer_type = type(layer)
        compile_layer=None
        if layer_type == nn.Conv2d:
            compile_layer=tikz_Conv("Conv2d",offset=f"({current_z},0,0)")
            current_z += z + spacing
            compiled_list.append(compile_layer)
        
    if to_file:
        tikz_save(to_file,compiled_list)
    # Start bulding image

    # for idx, (layer, shape) in enumerate(zip(layers_list, shapes)):

    #     if type(layer) in type_ignore or idx in index_ignore:
    #         continue

    #     layer_type = type(layer)

    #     if legend and show_dimension:
    #         layer_types.append(layer_type)
    #     elif layer_type not in layer_types:
    #         layer_types.append(layer_type)

        # x, y, z = calculate_layer_dimensions(
        #     shape, scale_z, scale_xy,
        #     max_z, max_xy, min_z, min_xy,
        #     one_dim_orientation, sizing_mode,
        #     dimension_caps, relative_base_size
        # )

    #     if legend and show_dimension:
    #         dimension_string = str(shape)
    #         dimension_string = dimension_string[1:len(
    #             dimension_string)-1].split(", ")
    #         dimension = []
    #         for i in range(0, len(dimension_string)):
    #             if dimension_string[i].isnumeric():
    #                 dimension.append(dimension_string[i])
    #         dimension_list.append(dimension)

    #     box = Box()
    #     box.de = 0
    #     if draw_volume and idx not in index_2d:
    #         box.de = x / 3
    #     if x_off == -1:
    #         x_off = box.de / 2

    #     box.x1 = current_z - box.de / 2
    #     box.y1 = box.de
    #     box.x2 = box.x1 + z
    #     box.y2 = box.y1 + y

    #     box.fill = color_map.get(layer_type, {}).get(
    #         'fill', color_wheel.get_color(layer_type))
    #     box.outline = color_map.get(layer_type, {}).get('outline', 'black')
    #     color_map[layer_type] = {'fill': box.fill, 'outline': box.outline}
    #     box.shade = shade_step

    #     boxes.append(box)
    #     layer_y.append(box.y2 - (box.y1 - box.de))
    #     hh = box.y2 - (box.y1 - box.de)
    #     img_height = max(img_height, hh)
    #     max_right = max(max_right, box.x2 + box.de)
    #     current_z += z + spacing

    # img_width = max_right + x_off + padding
    # img_height = max(img_height, hh)

    # is_any_text_above = False
    # is_any_text_below = False
    # max_box_with_text_height = 0
    # max_box_height = 0

    # if text_callable is not None:
    #     if font is None:
    #         font = ImageFont.load_default()
    #     i = -1
    #     for index, layer in enumerate(model.layers):
    #         if type(layer) in type_ignore or type(layer) == index in index_ignore:
    #             continue
    #         i += 1
    #         text, above = text_callable(i, layer)
    #         if above:
    #             is_any_text_above = True
    #         else:
    #             is_any_text_below = True

    #         text_height = 0
    #         for line in text.split('\n'):
    #             if hasattr(font, 'getsize'):
    #                 line_height = font.getsize(line)[1]
    #             else:
    #                 line_height = font.getbbox(line)[3]
    #             text_height += line_height
    #         text_height += (len(text.split('\n'))-1)*text_vspacing
    #         box_height = abs(boxes[i].y2-boxes[i].y1)-boxes[i].de
    #         box_with_text_height = box_height + text_height
    #         if box_with_text_height > max_box_with_text_height:
    #             max_box_with_text_height = box_with_text_height
    #         if box_height > max_box_height:
    #             max_box_height = box_height
    # if is_any_text_above:
    #     img_height += abs(max_box_height - max_box_with_text_height)*2

    # img = Image.new('RGBA', (int(ceil(img_width)),
    #                 int(ceil(img_height))), background_fill)

    # # x, y correction (centering)
    # for i, node in enumerate(boxes):
    #     y_off = (img.height - layer_y[i]) / 2
    #     node.y1 += y_off
    #     node.y2 += y_off

    #     node.x1 += x_off
    #     node.x2 += x_off

    # if is_any_text_above:
    #     img_height -= abs(max_box_height - max_box_with_text_height)
    #     img = Image.new('RGBA', (int(ceil(img_width)),
    #                     int(ceil(img_height))), background_fill)
    # if is_any_text_below:
    #     img_height += abs(max_box_height - max_box_with_text_height)
    #     img = Image.new('RGBA', (int(ceil(img_width)),
    #                     int(ceil(img_height))), background_fill)

    # draw = aggdraw.Draw(img)

    # # Correct x positions of reversed boxes
    # if draw_reversed:
    #     for box in boxes:
    #         offset = box.de
    #         # offset = 0
    #         box.x1 = box.x1 + offset
    #         box.x2 = box.x2 + offset

    # # Draw created boxes

    # last_box = None

    # if draw_reversed:
    #     for box in boxes:
    #         pen = aggdraw.Pen(get_rgba_tuple(box.outline))

    #         if last_box is not None and draw_funnel:
    #             # Top connection back
    #             draw.line([last_box.x2 - last_box.de, last_box.y1 - last_box.de,
    #                        box.x1 - box.de, box.y1 - box.de], pen)
    #             # Bottom connection back
    #             draw.line([last_box.x2 - last_box.de, last_box.y2 - last_box.de,
    #                        box.x1 - box.de, box.y2 - box.de], pen)

    #         last_box = box

    #     last_box = None

    #     for box in reversed(boxes):
    #         pen = aggdraw.Pen(get_rgba_tuple(box.outline))

    #         if last_box is not None and draw_funnel:
    #             # Top connection front
    #             draw.line([last_box.x1, last_box.y1,
    #                        box.x2, box.y1], pen)

    #             # Bottom connection front
    #             draw.line([last_box.x1, last_box.y2,
    #                        box.x2, box.y2], pen)

    #         box.draw(draw, draw_reversed=True)

    #         last_box = box
    # else:
    #     for box in boxes:
    #         pen = aggdraw.Pen(get_rgba_tuple(box.outline))

    #         if last_box is not None and draw_funnel:
    #             draw.line([last_box.x2 + last_box.de, last_box.y1 - last_box.de,
    #                        box.x1 + box.de, box.y1 - box.de], pen)
    #             draw.line([last_box.x2 + last_box.de, last_box.y2 - last_box.de,
    #                        box.x1 + box.de, box.y2 - box.de], pen)

    #             draw.line([last_box.x2, last_box.y2,
    #                        box.x1, box.y2], pen)

    #             draw.line([last_box.x2, last_box.y1,
    #                        box.x1, box.y1], pen)

    #         box.draw(draw, draw_reversed=False)

    #         last_box = box

    # draw.flush()

    # if text_callable is not None:
    #     draw_text = ImageDraw.Draw(img)
    #     i = -1
    #     for index, layer in enumerate(model.layers):
    #         if type(layer) in type_ignore or type(layer) == index in index_ignore:
    #             continue
    #         i += 1
    #         text, above = text_callable(i, layer)
    #         text_height = 0
    #         text_x_adjust = []
    #         for line in text.split('\n'):
    #             if hasattr(font, 'getsize'):
    #                 line_height = font.getsize(line)[1]
    #             else:
    #                 line_height = font.getbbox(line)[3]

    #             text_height += line_height

    #             if hasattr(font, 'getsize'):
    #                 text_x_adjust.append(font.getsize(line)[0])
    #             else:
    #                 text_x_adjust.append(font.getbbox(line)[2])
    #         text_height += (len(text.split('\n'))-1)*text_vspacing

    #         box = boxes[i]
    #         text_x = box.x1 + (box.x2 - box.x1) / 2
    #         text_y = box.y2
    #         if above:
    #             text_x = box.x1 + box.de + (box.x2 - box.x1) / 2
    #             text_y = box.y1 - box.de - text_height

    #         # Shift text to the left by half of the text width, so that it is centered
    #         text_x -= max(text_x_adjust)/2
    #         # Centering with middle text anchor 'm' does not work with align center
    #         anchor = 'la'
    #         if above:
    #             anchor = 'la'

    #         draw_text.multiline_text((text_x, text_y), text, font=font, fill=font_color,
    #                                  anchor=anchor, align='center',
    #                                  spacing=text_vspacing)

    # # Create layer color legend
    # if legend:
    #     if font is None:
    #         font = ImageFont.load_default()

    #     if hasattr(font, 'getsize'):
    #         text_height = font.getsize("Ag")[1]
    #     else:
    #         text_height = font.getbbox("Ag")[3]
    #     cube_size = text_height

    #     de = 0
    #     if draw_volume:
    #         de = cube_size // 2

    #     patches = list()

    #     if show_dimension:
    #         counter = 0

    #     for layer_type in layer_types:
    #         if show_dimension:
    #             label = layer_type.__name__ + \
    #                 "(" + str(dimension_list[counter]) + ")"
    #             counter += 1
    #         else:
    #             label = layer_type.__name__

    #         if hasattr(font, 'getsize'):
    #             text_size = font.getsize(label)
    #         else:
    #             # Get last two values of the bounding box
    #             # getbbox returns 4 dimensions in total, where the first two are always zero,
    #             # So we fetch the last two dimensions to match the behavior of getsize
    #             text_size = font.getbbox(label)[2:]
    #         label_patch_size = (2 * cube_size + de +
    #                             spacing + text_size[0], cube_size + de)

    #         # this only works if cube_size is bigger than text height

    #         img_box = Image.new('RGBA', label_patch_size, background_fill)
    #         img_text = Image.new('RGBA', label_patch_size, (0, 0, 0, 0))
    #         draw_box = aggdraw.Draw(img_box)
    #         draw_text = ImageDraw.Draw(img_text)

    #         box = Box()
    #         box.x1 = cube_size
    #         box.x2 = box.x1 + cube_size
    #         box.y1 = de
    #         box.y2 = box.y1 + cube_size
    #         box.de = de
    #         box.shade = shade_step
    #         box.fill = color_map.get(layer_type, {}).get('fill', "#000000")
    #         box.outline = color_map.get(
    #             layer_type, {}).get('outline', "#000000")
    #         box.draw(draw_box, draw_reversed)

    #         text_x = box.x2 + box.de + spacing
    #         # 2D center; use text_height and not the current label!
    #         text_y = (label_patch_size[1] - text_height) / 2
    #         draw_text.text((text_x, text_y), label, font=font, fill=font_color)

    #         draw_box.flush()
    #         img_box.paste(img_text, mask=img_text)
    #         patches.append(img_box)

    #     legend_image = linear_layout(patches, max_width=img.width, max_height=img.height, padding=padding,
    #                                  spacing=spacing,
    #                                  background_fill=background_fill, horizontal=True)
    #     img = vertical_image_concat(
    #         img, legend_image, background_fill=background_fill)

    # if to_file is not None:
    #     img.save(to_file)

    # return img
