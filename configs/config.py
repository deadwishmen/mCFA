"""Configuration constants, emotion categories, and normalization statistics for EMOTIC."""

EMOTIC_CATEGORIES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment", "Doubt/Confusion",
    "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue",
    "Fear", "Happiness", "Pain", "Peace", "Pleasure",
    "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]

CAT2IND = {emotion: idx for idx, emotion in enumerate(EMOTIC_CATEGORIES)}
IND2CAT = {idx: emotion for idx, emotion in enumerate(EMOTIC_CATEGORIES)}

# Normalization statistics for each branch
CONTEXT_MEAN = [0.4690646, 0.4407227, 0.40508908]
CONTEXT_STD = [0.2514227, 0.24312855, 0.24266963]

BODY_MEAN = [0.43832874, 0.3964344, 0.3706214]
BODY_STD = [0.24784276, 0.23621225, 0.2323653]

SWIN_BODY_MEAN = [0.485, 0.456, 0.406]
SWIN_BODY_STD = [0.229, 0.224, 0.225]

FACE_MEAN = [0.507395516207, 0.507395516207, 0.507395516207]
FACE_STD = [0.255128989415, 0.255128989415, 0.255128989415]


def get_norms(is_swin: bool = True):
    """Return (context_norm, body_norm, face_norm) pairs of [mean, std]."""
    context_norm = [CONTEXT_MEAN, CONTEXT_STD]
    body_norm = [SWIN_BODY_MEAN, SWIN_BODY_STD] if is_swin else [BODY_MEAN, BODY_STD]
    face_norm = [FACE_MEAN, FACE_STD]
    return context_norm, body_norm, face_norm
