"""Multi-Branch Feature Fusion modules (Concatenation, Spatial Attention, and Cross + Self Attention)."""

import torch
import torch.nn as nn


class FusionConcat(nn.Module):
    """Simple feature concatenation fusion model (Context + Body + Face)."""

    def __init__(
        self,
        num_context_features: int,
        num_body_features: int,
        num_face_features: int,
        is_vad_prediction: bool = False,
        use_selu: bool = True,
    ):
        super(FusionConcat, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.is_vad_prediction = is_vad_prediction
        self.use_selu = use_selu

        self.fc1 = nn.Linear(
            self.num_context_features + self.num_body_features + self.num_face_features, 256
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x_context, x_body, x_face):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)
        fuse_features = torch.cat((context_features, body_features, face_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.selu(fuse_out) if self.use_selu else self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        if not self.is_vad_prediction:
            return cat_out
        return cont_out


class FusionSpatialAttention(nn.Module):
    """Spatial attention fusion between facial feature vector and spatial context feature map (N, C, H, W)."""

    def __init__(
        self,
        num_context_features: int,
        num_body_features: int,
        num_face_features: int,
        shape_output_context: tuple,
        is_vad_prediction: bool = False,
    ):
        super(FusionSpatialAttention, self).__init__()
        self.N, self.C, self.H, self.W = shape_output_context
        self.is_vad_prediction = is_vad_prediction

        self.flatten = nn.Flatten()
        self.attention_fc1 = nn.Sequential(
            nn.Linear(self.C * self.H * self.W * 2, 128 * self.H * self.W * 2),
            nn.ReLU(),
            nn.BatchNorm1d(128 * self.H * self.W * 2),
            nn.Linear(128 * self.H * self.W * 2, self.H * self.W),
            nn.LogSoftmax(dim=1),
        )
        self.body_weight = nn.Sequential(
            nn.Linear(self.C, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.context_weight = nn.Sequential(
            nn.Linear(self.C, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.fc1 = nn.Linear((self.C * 2 + self.num_face_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face):
        face_vec_repeat = x_face.unsqueeze(2).repeat(1, 1, self.H * self.W)
        context_vec = x_context.view(-1, self.C, self.H * self.W)
        concat1 = torch.cat([face_vec_repeat, context_vec], dim=-1)
        concat1 = self.flatten(concat1)

        attention_weight = self.attention_fc1(concat1)
        attention_weight = attention_weight.view(-1, self.H * self.W, 1)

        context_vec = torch.matmul(context_vec, attention_weight)
        context_vec = context_vec.view(-1, self.C)

        body_vec = x_body.view(-1, self.num_body_features)
        face_vec = x_face.view(-1, self.num_face_features)

        fuse_features = torch.cat((context_vec, body_vec, face_vec), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        if not self.is_vad_prediction:
            return cat_out
        return cont_out


class FusionCrossSelfAttention(nn.Module):
    """
    Hierarchical Cross-Attention + Self-Attention Fusion across Context, Body, and Face branches.
    Matches the `Fusion` class used for pretrained SwinT + Attention checkpoints.
    """

    def __init__(
        self,
        num_context_features: int,
        num_body_features: int,
        num_face_features: int,
        is_swin: bool = True,
        is_vad_prediction: bool = False,
    ):
        super(FusionCrossSelfAttention, self).__init__()
        self.x = torch.tensor(num_face_features, dtype=torch.float32)
        self.dk = torch.sqrt(self.x)
        self.softmax = nn.Sequential(nn.Softmax(dim=1))
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.is_swin = is_swin
        self.is_vad_prediction = is_vad_prediction

        if self.is_swin:
            self.num_features = num_body_features
            self.fc_context = nn.Linear(num_context_features, self.num_features)
        else:
            self.num_features = num_face_features
            self.fc_context = nn.Linear(num_context_features, self.num_features)
            self.fc_body = nn.Linear(num_body_features, self.num_features)

        self.fc_att = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.fc1 = nn.Linear(self.num_features * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, x_context, x_body, x_face):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)

        if self.is_swin:
            body_vec_repeat = face_features.unsqueeze(2).repeat(
                1, 1, int(self.num_body_features / self.num_face_features)
            )
            face_features = body_vec_repeat.view(-1, self.num_body_features)
            context_features = self.fc_context(context_features)
        else:
            context_features = self.fc_context(context_features)
            body_features = self.fc_body(body_features)

        context_features = context_features.view(-1, self.num_features, 1)
        body_features = body_features.view(-1, self.num_features, 1)
        face_features = face_features.view(-1, self.num_features, 1)

        # Cross Attention
        context_body = context_features.transpose(1, 2) @ body_features
        context_face = context_features.transpose(1, 2) @ face_features
        context_in = torch.cat((context_body, context_face), 1)
        context_atten = self.softmax(context_in)

        body_context = body_features.transpose(1, 2) @ context_features
        body_face = body_features.transpose(1, 2) @ face_features
        body_in = torch.cat((body_context, body_face), 1)
        body_atten = self.softmax(body_in)

        face_context = face_features.transpose(1, 2) @ context_features
        face_body = face_features.transpose(1, 2) @ body_features
        face_in = torch.cat((face_context, face_body), 1)
        face_atten = self.softmax(face_in)

        context_features = torch.sum(context_features @ context_atten.transpose(1, 2), dim=2, keepdim=True)
        body_features = torch.sum(body_features @ body_atten.transpose(1, 2), dim=2, keepdim=True)
        face_features = torch.sum(face_features @ face_atten.transpose(1, 2), dim=2, keepdim=True)

        # Self Attention across branches
        context_features = context_features.view(-1, self.num_features)
        body_features = body_features.view(-1, self.num_features)
        face_features = face_features.view(-1, self.num_features)

        score_context = self.fc_att(context_features)
        score_body = self.fc_att(body_features)
        score_face = self.fc_att(face_features)

        score = torch.cat((score_context, score_body, score_face), 1)
        score = self.softmax(score)
        score = score.view(-1, score.shape[1], 1)

        context_features = context_features * score[:, 0]
        body_features = body_features * score[:, 1]
        face_features = face_features * score[:, 2]

        fuse_features = torch.cat((context_features, body_features, face_features), 1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)

        if not self.is_vad_prediction:
            return cat_out
        return cont_out


# Alias `Fusion` to `FusionCrossSelfAttention` so serialized `torch.save(model)` checkpoints unpickle cleanly
Fusion = FusionCrossSelfAttention


def build_fusion_model(
    fusion_type: str,
    num_context_features: int,
    num_body_features: int,
    num_face_features: int,
    is_swin: bool = True,
    is_vad_prediction: bool = False,
    shape_output_context: tuple = None,
):
    """Factory function for constructing the desired Fusion architecture."""
    fusion_type = fusion_type.lower()
    if fusion_type == "concat":
        return FusionConcat(
            num_context_features,
            num_body_features,
            num_face_features,
            is_vad_prediction=is_vad_prediction,
        )
    elif fusion_type == "spatial_attention":
        if shape_output_context is None:
            raise ValueError("`shape_output_context` (N, C, H, W) is required for 'spatial_attention'.")
        return FusionSpatialAttention(
            num_context_features,
            num_body_features,
            num_face_features,
            shape_output_context=shape_output_context,
            is_vad_prediction=is_vad_prediction,
        )
    elif fusion_type in ("cross_self_attention", "attention"):
        return FusionCrossSelfAttention(
            num_context_features,
            num_body_features,
            num_face_features,
            is_swin=is_swin,
            is_vad_prediction=is_vad_prediction,
        )
    else:
        raise ValueError(f"Unsupported fusion_type: {fusion_type}")
