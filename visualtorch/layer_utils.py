# PyTorch reimplementation of advanced Keras graph utilities for visualization
# Converts Keras-specific traversal and adjacency logic to PyTorch-friendly nn.Module traversal

import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from .utils import get_keys_by_value, self_multiply


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
    for i, dummy_layer in enumerate(output_layers):
        id_to_num_mapping[id(dummy_layer)] = len(id_to_num_mapping)
        # connect last real layer to dummy
        adj_matrix[-2, len(id_to_num_mapping)-1] = 1
    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    return isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)


def extract_primary_shape(layer_output_shape, layer_name=None):
    if isinstance(layer_output_shape, tuple):
        return layer_output_shape
    elif isinstance(layer_output_shape, list) and layer_output_shape:
        return layer_output_shape[0]
    else:
        return (1,)


def calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy, one_dim_orientation='y', sizing_mode='accurate', dimension_caps=None, relative_base_size=20):
    dims = [d for d in shape if isinstance(d, int) and d is not None]
    if not dims:
        return (min_xy, min_xy, min_z)
    x = min(max(dims[0] * scale_xy, min_xy),
            max_xy) if len(dims) > 0 else min_xy
    y = min(max(dims[1] * scale_xy, min_xy),
            max_xy) if len(dims) > 1 else min_xy
    z = min(max(dims[2] * scale_z, min_z), max_z) if len(dims) > 2 else min_z
    return (x, y, z)
