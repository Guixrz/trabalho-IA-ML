from __future__ import annotations
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# Estatísticas do ImageNet — padrão para pesos pré-treinados
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ApplyCLAHE:
    """
    Aplica o CLAHE (Contrast Limited Adaptive Histogram Equalization)
    isolando o canal de Luminosidade (LAB) para não distorcer as cores.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # Guardamos apenas os números (que o Python sabe empacotar perfeitamente)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        np_img = np.array(img)
        lab_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_img)

        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return Image.fromarray(rgb_clahe)


def build_transforms(size: int = 224, clahe: bool = False) -> transforms.Compose:
    """
    Pipeline de pré-processamento estático (sem Data Augmentation).
    Usado para extrair características determinísticas usando a DenseNet.
    """
    pipeline = [transforms.Resize((size, size))]

    if clahe:
        pipeline.append(ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)))

    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    return transforms.Compose(pipeline)