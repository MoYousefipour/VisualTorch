from torchvision.models import resnet18
import visualtorch

model = resnet18()
visualtorch.layered_view(
    model, to_file='resnet-18.tex', min_z=1,max_z=15, sizing_mode='logarithmic', legend=True,spacing=5,padding=0)
