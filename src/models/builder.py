import math

import torch.nn as nn

from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights, RetinaNetHead, RetinaNetClassificationHead
from torchvision.models.segmentation.fcn import fcn_resnet50, FCN_ResNet50_Weights, FCNHead 

def efficientnetb0_builder(num_classes):
    """
    Build Imagenet pretrained EfficientNet model with specified number of classes and model name
    :param num_classes: number of classes
    :return: EfficientNet model
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    init_range = 1.0 / math.sqrt(num_classes)
    nn.init.uniform_(model.classifier[1].weight, -init_range, init_range)
    nn.init.zeros_(model.classifier[1].bias)
    return model

def retinanet_builder(num_classes, trainable_backbone_layers = 3, **kwargs):
    """
    Build COCO pretrained RetinaNet model with ResNet50 backbone, with specified number of classes and trainable backbone layers
    :param num_classes: number of classes
    :param trainable_backbone_layers: number of trainable backbone layers (from 0 to 5)
    :param kwargs: additional arguments to torchvision.models.detection.retinanet.RetinaNet
    :return: RetinaNet model
    """
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
    model = retinanet_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers, **kwargs)
    
    # Just replace the classification head
    # model.head.classification_head = RetinaNetClassificationHead(in_channels=model.backbone.out_channels, num_anchors=model.anchor_generator.num_anchors_per_location()[0], num_classes=num_classes, norm_layer=nn.BatchNorm2d)
    model.head = RetinaNetHead(in_channels=model.backbone.out_channels, num_anchors=model.anchor_generator.num_anchors_per_location()[0], num_classes=num_classes)
    return model

def fcn_builder(num_classes, aux_loss=True):
    """
    Build COCO pretrained FCN model with ResNet50 backbone, with specified number of classes
    :param num_classes: number of classes
    :return: FCN model
    """
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights, aux_loss=aux_loss)
    model.aux_classifier = FCNHead(1024, num_classes)
    model.classifier = FCNHead(2048, num_classes)
    return model