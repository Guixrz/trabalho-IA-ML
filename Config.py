import os
from dataclasses import dataclass
from pathlib import Path


def bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "none")


# NOVA FUNÇÃO: Lê números decimais do terminal
def float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass
class ConfigClassicML:
    seed: int = 42
    img_size: int = 224
    num_classes: int = 2
    """ divisao dos dados
    70% treino / 30% teste
    50 / 50
    60 / 40
    40 / 60
    """
    test_size: float = 0.30

    # Pré-processamento
    clahe: bool = True

    # NOVO: Limiar dinâmico (Padrão inicial é 0.50)
    threshold: float = 0.90

    knn_neighbors: int = 5
    knn_weights: str = "distance"
    knn_metric: str = "euclidean"
    nb_var_smoothing: float = 1e-9

    dataset_root: str = "ROP dataset"
    images_dir: str = f"{dataset_root}/image"
    metadata_path: str = f"{dataset_root}/zip information.xlsx"
    results_dir: str = ""

    def __post_init__(self):
        self.clahe = bool_from_env("CLAHE", self.clahe)
        self.threshold = float_from_env("THRESHOLD", self.threshold)  # Atualiza o limiar dinamicamente

        flags = "ClassicML"
        if self.clahe:
            flags += "_CLAHE"

        self.results_dir = f"./results/{flags}/"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        if not Path(self.images_dir).exists():
            print(f"[AVISO] Diretório de imagens não encontrado: {self.images_dir}")
        if not Path(self.metadata_path).exists():
            print(f"[AVISO] Arquivo de metadados não encontrado: {self.metadata_path}")


cfg = ConfigClassicML()