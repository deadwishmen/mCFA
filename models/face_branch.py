"""Branch 3: Face Feature Extraction backbones (S-FER, B-FER, and VGG-FER)."""

from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


class Face_Emotion_CNN(nn.Module):
    """Original 7-class S-FER Convolutional Neural Network (1x48x48 grayscale input)."""

    def __init__(self):
        super(Face_Emotion_CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Face_Emotion_CNN_new(nn.Module):
    """S-FER feature extractor (strips `fc3` to output 256-dim facial emotion features)."""

    def __init__(self):
        super(Face_Emotion_CNN_new, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Net(nn.Module):
    """B-FER network trained on FER-2013 (3x48x48 input)."""

    def __init__(self, dropout: float = 1.0):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=7, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool3(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)
        x = x.view(-1, 7)
        return F.log_softmax(x, dim=-1)


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def build_vgg_face_torch():
    """Construct the VGG_Face_torch Sequential architecture."""
    return nn.Sequential(
        nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(25088, 4096)),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 4096)),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 2622)),
    )


VGG_Face_torch = build_vgg_face_torch()


class VGG_Net(nn.Module):
    """VGG-FER wrapper replacing the final 2622-way layer with a 7-class classifier."""

    def __init__(self, model):
        super(VGG_Net, self).__init__()
        self.pre_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        x = self.classifier(x)
        return x


def build_face_backbone(backbone_type: str = "sfer", checkpoint_path: str = None):
    """
    Build the Face Feature Extraction branch.

    Supported `backbone_type` values:
    - 'sfer': Face_Emotion_CNN_new (1x48x48 input, 256 output features, is_bfer=False)
    - 'bfer': Net without final convblock10 (3x48x48 input, 256 output features, is_bfer=True)
    - 'vgg_fer': VGG_Net wrapper around VGG_Face_torch (3x48x48 input)
    """
    backbone_type = backbone_type.lower()
    if backbone_type == "sfer":
        base_model = Face_Emotion_CNN()
        if checkpoint_path is not None:
            base_model.load_state_dict(
                torch.load(checkpoint_path, map_location=lambda storage, loc: storage),
                strict=False,
            )
        state_dict = base_model.state_dict()
        state_dict.pop("fc3.weight", None)
        state_dict.pop("fc3.bias", None)
        model_face = Face_Emotion_CNN_new()
        model_face.load_state_dict(state_dict)
        return model_face, 256, False

    elif backbone_type == "bfer":
        base_model = Net(1.0)
        if checkpoint_path is not None:
            base_model.load_state_dict(
                torch.load(checkpoint_path, map_location=lambda storage, loc: storage),
                strict=False,
            )
        model_face = nn.Sequential(*(list(base_model.children())[:-1]))
        return model_face, 256, True

    elif backbone_type == "vgg_fer":
        vgg_base = build_vgg_face_torch()
        if checkpoint_path is not None:
            vgg_base.load_state_dict(
                torch.load(checkpoint_path, map_location=lambda storage, loc: storage),
                strict=False,
            )
        model_face = VGG_Net(vgg_base)
        return model_face, 7, True

    else:
        raise ValueError(f"Unsupported face backbone_type: {backbone_type}")
