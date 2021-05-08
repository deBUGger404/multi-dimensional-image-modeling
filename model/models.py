import torch
import torch.nn as nn
from torchvision import  models

# Freeze parameters so we don't backprop through them
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

# models are pretrained on 3 dimension images, if we want for more than 3 dimension then we have to set random weights for remain dimesions
def weights_initialize(model, input_dim):
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = weight
        model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
        model.conv1.weight[:, 4] = model.conv1.weight[:, 0]
        model.conv1.weight[:, 5] = model.conv1.weight[:, 0]
        model.conv1.weight[:, 6] = model.conv1.weight[:, 0]
        model.conv1.weight[:, 7] = model.conv1.weight[:, 0]
    return model

# load pretrained model for 3 dimension(rgb) model and assign random weights for remain dimension
def initialize_model(model_name, num_classes, input_dim, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft)
        if input_dim > 3:
            model_ft = weights_initialize(model_ft, input_dim)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, num_classes))
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if input_dim > 3:
            model_ft = weights_initialize(model_ft, input_dim)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, num_classes))
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if input_dim > 3:
            model_ft = weights_initialize(model_ft, input_dim)
        num_ftrs = model_ft.classifier.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, num_classes))
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if input_dim > 3:
            model_ft = weights_initialize(model_ft, input_dim)
        num_ftrs = model_ft.classifier.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, num_classes))
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size
