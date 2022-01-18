from torchvision.models import resnet18
import torch


model = resnet18(pretrained=True)

script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

