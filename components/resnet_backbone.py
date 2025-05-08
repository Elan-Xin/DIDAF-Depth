import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet18, resnet34, resnet50

# 其他代码保持不变


_MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

def load_pretrained_weights(name: str, map_location: str):
    assert name in _MODEL_URLS, f'无法找到{name}对应的URL。'
    state_dict = load_state_dict_from_url(_MODEL_URLS[name], map_location=map_location)
    return state_dict

def resnet18_backbone(pre_trained=False, **kwargs):
    model = resnet18(pretrained=pre_trained, **kwargs)
    return model

def resnet34_backbone(pre_trained=False, **kwargs):
    model = resnet34(pretrained=pre_trained, **kwargs)
    return model

def resnet50_backbone(pre_trained=False, **kwargs):
    model = resnet50(pretrained=pre_trained, **kwargs)
    return model
