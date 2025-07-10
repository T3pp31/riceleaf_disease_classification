
import torch
import torch.nn as nn
import timm

import config

def build_model(model_name='efficientnet_b0', pretrained=True, freeze_base=True):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROP_RATE if hasattr(config, 'DROP_RATE') else 0.2,
        drop_path_rate=config.DROP_PATH_RATE if hasattr(config, 'DROP_PATH_RATE') else 0.2
    )
    
    if freeze_base:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    return model

def get_available_models():
    recommended_models = [
        'efficientnet_b0',
        'efficientnet_b1', 
        'efficientnet_b2',
        'efficientnet_b3',
        'convnext_tiny',
        'convnext_small',
        'vit_small_patch16_224',
        'swin_tiny_patch4_window7_224',
        'resnet50',
        'densenet121'
    ]
    return recommended_models
