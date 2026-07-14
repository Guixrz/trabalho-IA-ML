"""
FeatureExtractorDenseNet.py

Extração de características profundas usando DenseNet-121 pré-treinada na ImageNet.
Pode ser usada como Drop-in replacement para a extração clássica, gerando
embeddings de 1024 dimensões.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class DenseNetFeatureExtractor:
    def __init__(self, fine_tuned_weights_path: str = None, device: str = None):
        """
        Inicializa a DenseNet para extração de características.
        
        Parâmetros:
        - fine_tuned_weights_path: Caminho para os pesos `.pth` se você fez fine-tuning prévio 
                                   no seu dataset de ROP. Se None, usa pesos da ImageNet.
        - device: 'cuda' ou 'cpu'. Se None, detecta automaticamente.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DenseNet Extractor] Inicializando no dispositivo: {self.device}")

        # 1. Carrega o modelo base
        if fine_tuned_weights_path:
            # Se for carregar pesos finetunados (após treinar no dataset ROP)
            print(f"[DenseNet Extractor] Carregando pesos fine-tuned de: {fine_tuned_weights_path}")
            self.model = models.densenet121(weights=None)
            
            # Ajuste caso o fine-tuning tenha mudado o número de classes (ex: 2 para binário)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, 2)
            
            self.model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=self.device))
        else:
            # Drop-in direto dos pesos da ImageNet
            print("[DenseNet Extractor] Usando pesos padrão da IMAGENET1K_V1")
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # 2. Remove o classificador final para extrair apenas as características (features)
        # O self.model.features já retorna as saídas convolucionais da DenseNet
        self.feature_extractor = self.model.features
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()  # Modo de avaliação (desativa Dropout/BatchNorm)

        # 3. Pooling Global e Achatamento
        # A DenseNet retorna tensores (Batch, 1024, 7, 7). Precisamos reduzir para (1024,)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Pipeline de transformações obrigatório da ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalização padrão exigida para modelos pré-treinados no PyTorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path: str) -> np.ndarray:
        """
        Lê uma imagem do disco, passa pela DenseNet e retorna o vetor de 1024 dimensões.
        """
        try:
            # Garante que a imagem está em RGB (ignora canal alfa de PNGs, por exemplo)
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Erro ao carregar a imagem {image_path}: {e}")

        # Aplica os transforms e adiciona dimensão de batch -> shape: (1, 3, 224, 224)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad(): # Desliga o rastreamento de gradientes para economizar memória
            # Passa pelas camadas da DenseNet
            features = self.feature_extractor(input_tensor)
            
            # Aplica ReLU (padrão da DenseNet antes do pooling final)
            features = nn.functional.relu(features, inplace=True)
            
            # Aplica Average Pooling e achata o tensor -> shape: (1, 1024)
            pooled_features = self.global_pool(features)
            flattened_features = torch.flatten(pooled_features, 1)

        # Move de volta para CPU e converte para array numpy (shape: 1024,)
        return flattened_features.cpu().numpy().squeeze()