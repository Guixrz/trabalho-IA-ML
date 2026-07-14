import cv2 # cvgen
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path


class FeatureExtractor:
    """
    Classe para extrair características estáticas de imagens para algoritmos clássicos de AM.
    Extrai:
    - Médias e Desvios Padrões de Cor (Canais RGB)
    - Estatísticas de Textura GLCM (Contraste, Correlação, Energia, Homogeneidade)
    """

    def __init__(self, apply_clahe: bool = False):
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _process_clahe(self, img_rgb: np.ndarray) -> np.ndarray:
        # Aplica CLAHE apenas no canal L (Luminosidade) do espaço LAB
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def extract(self, image_path: str) -> np.ndarray:
        # Carregamento e Pré-processamento Base
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Erro ao carregar a imagem: {image_path}")

        # cores
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionamento padrão para garantir que o tempo de cálculo da GLCM seja uniforme
        img_rgb = cv2.resize(img_rgb, (224, 224))

        if self.apply_clahe:
            img_rgb = self._process_clahe(img_rgb)

        features = []

        # CARACTERÍSTICAS DE COR (6 features)
        # Média e Desvio Padrão para R, G e B
        means = np.mean(img_rgb, axis=(0, 1))
        stds = np.std(img_rgb, axis=(0, 1))
        features.extend(means)
        features.extend(stds)

        # CARACTERÍSTICAS DE TEXTURA GLCM (4 features)
        # Converte para tons de cinza para analisar a textura
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Calcula a Matriz de Co-ocorrência (GLCM)
        # distances=[1] analisa pixels adjacentes. angles=[0] analisa na horizontal.
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        features.extend([contrast, correlation, energy, homogeneity])

        # Retorna um vetor 1D com 10 características (6 de cor + 4 de textura)
        return np.array(features, dtype=np.float32)