import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from efficientnet_pytorch import EfficientNet
from model_utils.attention_modules import Spatial_Attention
from model_utils.attention_modules import Temporal_Attention
from detr2 import DETR


class MYNET(nn.Module):
    def __init__(self, sequence_size):
        super().__init__()
        self.sequence_size = sequence_size
        # self.SA = Spatial_Attention(sequence_size+1)
        self.SA = Spatial_Attention(sequence_size)
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)

        self.TA = Temporal_Attention(n_segment=sequence_size//3, feature_dim=1000, num_class=2)

        self.anomaly_model = DETR(1)

    def forward(self, rgb):
        B = rgb.shape[0]
        SA = self.SA(rgb)
        encoded_features = rgb[:,1:,...]*SA

        # encoded_featurestorch.cat((encoded_features[:, 0, :].unsqueeze(1), encoded_features), 1)

        backbone_out = self.backbone(encoded_features.mean(2).reshape(-1, 3, 224, 224))

        features = self.backbone.extract_features(encoded_features.mean(2).reshape(-1, 3, 224, 224))
        # endpoints = self.backbone.extract_endpoints(encoded_features.mean(2).reshape(-1, 3, 224, 224))
        # features = endpoints['reduction_3']
        # (B,10,1280,7,7)

        output = self.anomaly_model(features.view(-1, self.sequence_size//3, features.shape[1],features.shape[2], features.shape[3]))

        output = output.reshape(B, self.sequence_size//3, -1)

        temporal_vec = (backbone_out).reshape(B, self.sequence_size//3, -1)

        temporal_vec = torch.mul(temporal_vec, output)

        output = self.TA(temporal_vec)
        return output
