from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
from src import resnet_models

def Detector(MODEL_SELECT, NUM_SPOOF_CLASS):

    if MODEL_SELECT == 0:
        print('using ResNet34.')
        model = resnet_models.resnet34(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 1:
        print('using SEResNet34.')
        model = resnet_models.se_resnet34(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 2:
        print('using ResNet50.')
        model = resnet_models.resnet50(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 3:
        print('using SEResNet50.')
        model = resnet_models.se_resnet50(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 4:
        print('using Res2Net50_26w_4s.')
        model = resnet_models.res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 5:
        print('using SERes2Net50_26w_4s.')
        model = resnet_models.se_res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 6:
        print('using Res2Net50_14w_8s.')
        model = resnet_models.res2net50_v1b_14w_8s(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 7:
        print('using SERes2Net50_14w_8s.')
        model = resnet_models.se_res2net50_v1b_14w_8s(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 8:
        print('using Res2Net50_26w_8s.')
        model = resnet_models.res2net50_v1b_26w_8s(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 9:
        print('using SERes2Net50_26w_8s.')
        model = resnet_models.se_res2net50_v1b_26w_8s(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    return model 

def test_Detector(model_id=0):
    model_params = {
        "MODEL_SELECT" : model_id,
        "NUM_SPOOF_CLASS" : 2
    }

    model = Detector(**model_params)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model contains {} parameters'.format(model_params))
    # print(model)
    x = torch.randn(2,1,257,400)
    output = model(x)
    print(x.size())
    print(output.size())
    
if __name__ == '__main__':
    for id in range(10):
        test_Detector(id)


