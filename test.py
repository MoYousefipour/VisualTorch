from torchvision.models import resnet18
import visualtorch

model = resnet18()
visualtorch.layered_view(
    model, to_file='resnet-18.png', min_z=3, sizing_mode='logarithmic', legend=True)
