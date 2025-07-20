import numpy as np
import torch.nn as nn
from .utils import self_multiply
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


def calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy, one_dim_orientation='y', sizing_mode='accurate', dimension_caps=None, relative_base_size=20):
    dims = tuple([d for d in shape if isinstance(d, int) and d is not None])
    if not dims:
        return min_xy, min_xy, min_z

    channel_cap = dimension_caps.get(
        'channels', max_z) if dimension_caps else max_z
    sequence_cap = dimension_caps.get(
        'sequence', max_xy) if dimension_caps else max_xy

    def smart_scale(value, base_scale, min_val, cap_val):
        """
        Smart scaling that maintains relative proportions while preventing extremely large visualizations.
        This provides the relative scaling functionality that was promised by the relative_scaling parameter.
        """
        if value <= 64:
            # Small dimensions: use full scaling to make them visible
            return min(max(value * base_scale, min_val), cap_val)
        elif value <= 512:
            # Medium dimensions: reduce scaling to balance visibility and proportion
            return min(max(value * base_scale * 0.6, min_val), cap_val)
        elif value <= 2048:
            # Large dimensions: further reduce scaling but maintain relative differences
            return min(max(value * base_scale * 0.3, min_val), cap_val)
        else:
            # Very large dimensions: use logarithmic scaling but still maintain relativity
            rel_log_scale = math.log10(value) * base_scale * 15
            return min(max(rel_log_scale, min_val), cap_val)

    def log_scale(value, base_scale, min_val, cap_val):
        if value <= 1:
            return min_val
        log_val = math.log10(value) * base_scale * 20
        return min(max(log_val, min_val), cap_val)

        # Accurate mode: mirror original block
    if sizing_mode == 'accurate':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = min(max(dims[0] * scale_xy, min_xy), max_xy)
                return min_xy, y, min_z
            else:
                z = min(max(dims[0] * scale_z, min_z), max_z)
                return min_xy, min_xy, z
        elif len(dims) == 2:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(dims[1] * scale_z, min_z), max_z)
            return x, y, z
        else:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(self_multiply(dims[2:]) * scale_z, min_z), max_z)
            return x, y, z

    # Capped mode
    elif sizing_mode == 'capped':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
                return min_xy, y, min_z
            else:
                z = max(min(dims[0] * scale_z, channel_cap), min_z)
                return min_xy, min_xy, z
        elif len(dims) == 2:
            x = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
            y = max(min(dims[1] * scale_xy, sequence_cap), min_xy)
            z = max(min(dims[2] * scale_z if len(dims)
                    > 2 else min_z, channel_cap), min_z)
            return x, y, z

    # Balanced mode
    elif sizing_mode == 'balanced':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return min_xy, int(y), min_z
            else:
                z = smart_scale(dims[0], scale_z, min_z, channel_cap)
                return min_xy, min_xy, int(z)
        else:
            x = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = smart_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = smart_scale(dims[2] if len(dims) > 2 else 1,
                            scale_z, min_z, channel_cap)
            return int(x), int(y), int(z)

    # Logarithmic mode
    elif sizing_mode == 'logarithmic':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return min_xy, int(y), min_z
            else:
                z = log_scale(dims[0], scale_z, min_z, channel_cap)
                return min_xy, min_xy, int(z)
        else:
            x = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = log_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = log_scale(dims[2] if len(dims) > 2 else 1,
                          scale_z, min_z, channel_cap)
            return int(x), int(y), int(z)

    # Relative mode - True proportional scaling where each layer's size is directly proportional to its dimension
    elif sizing_mode == 'relative':
        def proportional_scale(dimension, rel_base_size, min_val, max_val):
            """
            Scale dimension proportionally where relative_base_size represents 
            the visual size for dimension=1.

            Args:
                dimension (int): The dimension value to scale
                rel_base_size (int): Visual size (in pixels) for dimension=1
                min_val (int): Minimum allowed scaled value
                max_val (int): Maximum allowed scaled value

            Returns:
                int: Scaled dimension with true proportional relationships

            Formula: visual_size = dimension * relative_base_size
            """
            if dimension <= 0:
                return min_val

            # True proportional scaling: dimension * base_size
            scaled = dimension * rel_base_size

            # Apply min/max constraints while preserving proportionality as much as possible
            return max(min_val, min(scaled, max_val))

        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = proportional_scale(
                    dims[0], relative_base_size, min_xy, sequence_cap)
                return min_xy, y, min_z
            else:
                z = proportional_scale(
                    dims[0], relative_base_size, min_z, channel_cap)
                return min_xy, min_xy, z
        elif len(dims) == 2:
            x = proportional_scale(
                dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(
                dims[1], relative_base_size, min_xy, sequence_cap)

            # For 2D layers, use the second dimension for z-scaling as well
            z = proportional_scale(
                dims[1], relative_base_size, min_z, channel_cap)
            return x, y, z
        else:
            # 3D+ layers: handle spatial dimensions and channels separately
            x = proportional_scale(
                dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(
                dims[1], relative_base_size, min_xy, sequence_cap)

            # For channels (typically dims[2:]), use product for z-dimension
            channel_product = self_multiply(dims[2:]) if len(dims) > 2 else 1
            z = proportional_scale(
                channel_product, relative_base_size, min_z, channel_cap)
            return x, y, z
    else:
        warnings.warn(
            f"Unknown sizing mode '{sizing_mode}'. Defaulting to accurate.",
            UserWarning,
            stacklevel=3
        )

        # Recursive call to handle accurate sizing
        return calculate_layer_dimensions(
            shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
            one_dim_orientation=one_dim_orientation, sizing_mode='accurate',
            dimension_caps=dimension_caps, relative_base_size=relative_base_size
        )

