import os
from dataclasses import dataclass
from pathlib import Path


def bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "none")


@dataclass
class ConfigClassicML:
    # configurações gerais
    seed: int = 42
    img_size: int = 224
    num_classes: int = 2

    """ divisao dos dados
    70% treino  / 30% teste
    50          / 50
    60          / 40
    40          / 60
    """
    test_size: float = 0.30

    # pre processamento estatico
    clahe: bool = True

    # Data Augmentation dinâmico e Reinhard foram removidos do escopo
    # clássico para manter a extração determinística.

    # hiperparametros do KNN
    knn_neighbors: int = 5
    knn_weights: str = "distance"  # 'uniform' ou 'distance' (útil para desbalanceamento)
    knn_metric: str = "euclidean"  # 'euclidean' ou 'manhattan'

    # hiperparametros do naive bayes
    # var_smoothing resolve o problema da variância zero na distribuição Gaussiana[cite: 1]
    nb_var_smoothing: float = 1e-9

    # diretorios
    dataset_root: str = "ROP dataset"
    images_dir: str = f"{dataset_root}/image"
    metadata_path: str = f"{dataset_root}/zip information.xlsx"
    results_dir: str = ""  # resultados

    def __post_init__(self):
        self.clahe = bool_from_env("CLAHE", self.clahe)

        # Montagem do nome da pasta de resultados baseada nas flags ativas
        flags = "ClassicML"
        if self.clahe:
            flags += "_CLAHE"

        self.results_dir = f"./results/{flags}/"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        # Avisos de verificação
        if not Path(self.images_dir).exists():
            print(f"[AVISO] Diretório de imagens não encontrado: {self.images_dir}")
        if not Path(self.metadata_path).exists():
            print(f"[AVISO] Arquivo de metadados não encontrado: {self.metadata_path}")


cfg = ConfigClassicML()