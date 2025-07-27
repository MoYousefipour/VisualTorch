from typing import Callable,Any
from PIL import Image, ImageDraw,ImageFont
from math import ceil
import torch
import torch.nn as nn


from .utils import *
from .layer_utils import *


def layered_view(model: nn.Module,
                 input_shape: tuple = None,
                 to_file: str = None,
                 min_z: int = 1,
                 min_xy: int = 1,
                 max_z: int = 50,
                 max_xy: int = 50,
                 scale_z: float = 2,
                 scale_xy: float = 6,
                 type_ignore: list = None,
                 index_ignore: list = None,
                 color_map: dict = None,
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
        # if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d, nn.Flatten)):

    # Run dummy forward pass to activate hooks and collect shapes
    try:
        # Create a dummy input tensor with batch size 1 matching model input
        next(model.modules())
        if hasattr(model, 'forward'):
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
    # color_wheel = ColorWheel()
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
    drawline_list=[]
    # Start Building Tikz
    for idx , (layer,shape) in enumerate(zip(layers_list,shapes)):
        if type(layer) in type_ignore or idx in index_ignore:
            continue
        x, y, z = calculate_layer_dimensions(
            shape, scale_z, scale_xy,
            max_z, max_xy, min_z, min_xy,
            dimension_caps, relative_base_size
        )
        
        layer_type = type(layer)
        compile_layer,to=None,None
        if layer_type == nn.Conv2d:
            compile_layer=tikz_Conv(f"conv2d-{idx}",n_filer=shape[1],s_filer=shape[-1],offset=f"({current_z},0,0)",height=y,depth=x,width=z,caption=layer_type.__name__)
            to = f"conv2d-{idx}"
        elif layer_type == nn.MaxPool2d:
            compile_layer=tikz_Pool(f"maxpool2d-{idx}",offset=f"({current_z},0,0)",height=y,depth=x,caption=layer_type.__name__)
            to= f"maxpool2d-{idx}"
        elif layer_type == nn.Linear:
            compile_layer = tikz_Fc(f"fc-linear-{idx}",n_filer=shape[1],s_filer=shape[-1],offset=f"({current_z},0,0)",depth=x,caption=layer_type.__name__)
            to= f"fc-linear-{idx}"
        elif layer_type == nn.Softmax:
            compile_layer = tikz_ConvSoftMax(f"softmax-{idx}",s_filer=shape[-1],offset=f"({current_z},0,0)",caption=layer_type.__name__)
            to= f"softmax-{idx}"
        elif layer_type == nn.AdaptiveAvgPool2d:
            compile_layer=tikz_Pool(f"adaptivepool2d-{idx}",offset=f"({current_z},0,0)",height=y,depth=x,caption=layer_type.__name__)
            to = f"adaptivepool2d-{idx}"
        elif layer_type == nn.BatchNorm2d:
            compile_layer=tikz_Pool(f"batchnorm2d-{idx}",offset=f"({current_z},0,0)",height=y,depth=x,caption=layer_type.__name__)
            to = f"batchnorm2d-{idx}"
        elif layer_type == nn.ReLU:
            compile_layer=tikz_Relu(f"relu-{idx}",offset=f"({current_z},0,0)",height=y,depth=x,caption=layer_type.__name__)
            to = f"relu-{idx}"
        if compile_layer:
            if len(compiled_list)>0:
                of=compiled_list[-1].split('name=')[1].split(',')[0]
                draw=tikz_connection(of,to)
                drawline_list.append(draw)
            compiled_list.append(compile_layer)
            current_z += z/8+spacing
        
    if to_file:
        tikz_save(to_file,compiled_list,drawline_list)