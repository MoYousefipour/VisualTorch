import numpy as np
import torch.nn as nn
from .utils import min_max
import math
import warnings


def get_incoming_layers(layer, parent_model):
    """Get previous layers connected to the given layer by sequential order."""
    layers = list(parent_model.children())
    idx = layers.index(layer)
    if idx == 0:
        return []
    return [layers[idx - 1]]


def get_outgoing_layers(layer, parent_model):
    """Get next layers connected to the given layer by sequential order."""
    layers = list(parent_model.children())
    idx = layers.index(layer)
    if idx == len(layers) - 1:
        return []
    return [layers[idx + 1]]


def model_to_adj_matrix(model):
    layers = list(model.children())
    adj_matrix = np.zeros((len(layers), len(layers)))
    id_to_num_mapping = {id(layer): idx for idx, layer in enumerate(layers)}
    for idx, layer in enumerate(layers[:-1]):
        adj_matrix[idx, idx + 1] = 1
    return id_to_num_mapping, adj_matrix


def find_layer_by_id(model, _id):
    for layer in model.children():
        if id(layer) == _id:
            return layer
    return None


def find_layer_by_name(model, name):
    for layer in model.named_children():
        if layer[0] == name:
            return layer[1]
    return None


def find_input_layers(model):
    layers = list(model.children())
    if layers:
        yield layers[0]


def find_output_layers(model):
    layers = list(model.children())
    if layers:
        yield layers[-1]


def model_to_hierarchy_lists(model):
    layers = list(model.children())
    return [[layer] for layer in layers]


def augment_output_layers(model, output_layers, id_to_num_mapping, adj_matrix):
    adj_matrix = np.pad(adj_matrix, ((0, len(output_layers)), (0, len(
        output_layers))), mode='constant', constant_values=0)
    for dummy_layer in output_layers:
        id_to_num_mapping[id(dummy_layer)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(output_layers[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    return isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)


def extract_primary_shape(layer_output_shape, layer_name=None):
    # Handle None or empty cases
    layer_info = f" (layer: {layer_name})" if layer_name else ""

    # Handle None case - warn and use default
    if layer_output_shape is None:
        warnings.warn(
            f"Layer output shape is None{layer_info}. This may indicate a demolish model "
            f"or invalid layer configuration. Using default shape (None, 1) for visualization.",
            UserWarning,
            stacklevel=3
        )
        return None, 1

    # Handle tuple case
    if isinstance(layer_output_shape, tuple):
        # Check if this is a multi-output scenario (tuple of tuples)
        if len(layer_output_shape) > 0 and isinstance(layer_output_shape[0], (tuple, list)):
            # Multi-output case
            warnings.warn(
                f"Multi-output layer detected{layer_info}. "
                f"Using primary output shape {layer_output_shape[0]} for visualization. "
                f"Secondary outputs {layer_output_shape[1:]} will be ignored.",
                UserWarning,
                stacklevel=3
            )
            return layer_output_shape[0]
        else:
            # Single output tuple
            return layer_output_shape

    # Handle list case
    elif isinstance(layer_output_shape, list):
        if len(layer_output_shape) == 1:
            # Single output
            return layer_output_shape[0]
        elif len(layer_output_shape) > 1:
            # Multi-output list
            warnings.warn(
                f"Multi-output layer detected{layer_info}. "
                f"Using primary output shape {layer_output_shape[0]} for visualization. "
                f"Secondary outputs {layer_output_shape[1:]} will be ignored.",
                UserWarning,
                stacklevel=3
            )
            return layer_output_shape[0]
        else:
            # Empty list
            warnings.warn(
                f"Layer output shape is an empty list{layer_info}. This indicates an invalid "
                f"layer configuration. Using default shape (None, 1) for visualization.",
                UserWarning,
                stacklevel=3
            )
            return None, 1

    # Unsupported format
    else:
        raise RuntimeError(
            f"Unsupported tensor shape format: {type(layer_output_shape).__name__} = {layer_output_shape}{layer_info}. "
            f"Expected tuple or list, but got {type(layer_output_shape).__name__}."
        )


def calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy, dimension_caps=None, relative_base_size=20):
    dims = tuple([d for d in shape if isinstance(d, int) and d is not None])
    if not dims:
        return min_xy, min_xy, min_z

    channel_cap = dimension_caps.get(
        'channels', max_z) if dimension_caps else max_z
    sequence_cap = dimension_caps.get(
        'sequence', max_xy) if dimension_caps else max_xy
    
    
    
    x=min_xy
    y=min_xy
    z=min_z
    
    if len(dims) == 1:
        z=min_max(dims[0], scale_z, min_z, channel_cap)
    elif len(dims) == 2:
        x=min_max(dims[1], scale_xy, min_xy, sequence_cap)
        y=min_max(dims[0], scale_xy, min_xy, sequence_cap)
        z=min_max(dims[0], scale_z, min_z, channel_cap)
    else:
        x=min_max(dims[2], scale_xy, min_xy, sequence_cap)
        y=min_max(dims[3], scale_xy, min_xy, sequence_cap)
        z=min_max(dims[1], scale_z, min_z, channel_cap)
    return x,y,z