
import torch
from .MiniFASNetV1SE import MiniFASNetV1SE
from collections import OrderedDict

class SilentFaceModel:
    def __init__(self, model_path, input_size=(80, 80)):
        self.input_width, self.input_height = input_size
        self.model = MiniFASNetV1SE(num_classes=2, input_size=input_size, width_mult=1.0)
        state_dict = torch.load(model_path, map_location='cpu')

        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def __call__(self, image_tensor):
        return self.model(image_tensor)
