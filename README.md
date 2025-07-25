# VisualTorch
Visualtorch is a Python package to help visualize torch (either standalone or included in pytorch) neural network architectures. It allows easy styling to fit most needs. This module supports layered style architecture generation which is great for CNNs (Convolutional Neural Networks), and a graph style architecture, which works great for most models including plain feed-forward networks. For help in citing this project, refer here.

This work is inspired from "paulgavrikov/visualkeras" which is makeing layers from Tensorflow and Keras, and using styles of "HarisIqbal88/PlotNeuralNet" to make it ready to work.

 **Citation**

If you find this project helpful for your research please consider citing it in your publication as follows.
```
@software{Yousefipour_VisualTorch_2025,
author = {Yousefipour, Mohammad},
license = {MIT},
month = jul,
title = {{VisualTorch}},
url = {https://github.com/MoYousefipour/VisualTorch},
version = {0.1.0},
year = {2025}
}
```
## How to Use
Clone the repository.

```terminal
clone https://github.com/MoYousefipour/VisualTorch.git
```
Change `test.py` file with you model that written with torch for example for vgg16:

```python
from torchvision.models import vgg16
import visualtorch
import torch.nn as nn
model = vgg16()
visualtorch.layered_view(
    model,input_shape=(1,3,224,224),to_file='vgg16.tex',spacing=2,padding=1)
```
Run `test.py` file, then compile the `.tex` file that given to see vgg16 architecture.

see <a href="https://github.com/MoYousefipour/VisualTorch/tree/master/example/vgg16"> VGG16 example!</a> 
