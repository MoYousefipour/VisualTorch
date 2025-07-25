from torchvision.models import vgg16
import visualtorch
import torch.nn as nn
model = vgg16()
visualtorch.layered_view(
    model,input_shape=(1,3,224,224),to_file='vgg16.tex',spacing=2,padding=1)
