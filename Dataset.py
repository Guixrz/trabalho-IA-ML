"""
Dataset.py — Manipulação do dataset Zhao.

Filosofia atualizada:
- O split dos dados (Treino/Teste) é feito a nível de PACIENTE para evitar data leakage.

Classificação Binária:
- Negativo (0): Normal (DG 0)
- Positivo (1): Estágios ativos da doença (DG 1, 2, 3, 4, 5)
- Excluídos: A-ROP e Laser-treated ROP (DG 6, 7)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from FeatureExtractor import FeatureExtractor

# Constantes de Diagnóstico
DG_NEGATIVE: frozenset[int] = frozenset({0})
DG_POSITIVE: frozenset[int] = frozenset({1, 2, 3, 4, 5})
DG_EXCLUDED: frozenset[int] = frozenset({6, 7})

DG_CLASS_NAMES_ZHAO: dict[str, int] = {
    "Normal": 0,
    "Stage1": 1,
    "Stage2": 2,
    "Stage3": 3,
    "laser scars": 7,
}


def build_manifest_zhao(metadata_path: str, images_dir: str) -> pd.DataFrame:
    """
    DataFrame especificamente para o Dataset de Zhao.
    Faz o vínculo PERFEITO entre a imagem na pasta e o ID do paciente na planilha.
    """
    metadata_path = Path(metadata_path)
    images_dir = Path(images_dir)

    df_meta = pd.read_excel(metadata_path, dtype={"ID": str})
    df_meta.columns = df_meta.columns.str.strip()

    if "img_name" not in df_meta.columns:
        print("Buscando mapeamento de imagens na segunda aba da planilha...")
        df_meta = pd.read_excel(metadata_path, sheet_name=1, dtype={"ID": str})
        df_meta.columns = df_meta.columns.str.strip()

    if "img_name" not in df_meta.columns or "ID" not in df_meta.columns:
        raise ValueError("A planilha não contém as colunas 'img_name' e 'ID' necessárias.")

    img_to_patient = dict(zip(df_meta["img_name"], df_meta["ID"]))

    records: list[dict] = []

    for folder_path in images_dir.iterdir():
        if not folder_path.is_dir() or folder_path.name not in DG_CLASS_NAMES_ZHAO:
            continue

        dg_code = DG_CLASS_NAMES_ZHAO[folder_path.name]

        if dg_code in DG_EXCLUDED:
            continue

        binary_label = 1 if dg_code in DG_POSITIVE else 0

        for img_path in folder_path.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img_name = img_path.name

            patient_id = img_to_patient.get(img_name)

            if pd.isna(patient_id) or patient_id is None:
                continue

            exam_id = f"{patient_id}_ex"

            records.append({
                "image_path": str(img_path),
                "image": img_name,
                "patient_id": str(patient_id),
                "exam_id": exam_id,
                "dg_code": dg_code,
                "binary_label": binary_label,
            })

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError(f"Nenhuma imagem válida com ID correspondente encontrada em: {images_dir}")

    return manifest.reset_index(drop=True)


DG_CLASS_NAMES: dict[str, int] = {
    "Normal": 0,
    "Stage 1 ROP": 1,
    "Stage 2 ROP": 2,
    "Stage 3 ROP": 3,
    "Laser-treated ROP": 7,
}


def build_tabular_dataset(df: pd.DataFrame, extractor: FeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recebe um DataFrame de metadados e um extrator.
    Retorna X (matriz de características) e y (vetor de rótulos binários).
    """
    X_list = []
    y_list = []

    print(f"Extraindo características de {len(df)} imagens...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"]
        label = row["binary_label"]

        try:
            # Chama o método que gera o vetor de 10 dimensões (cor + textura)
            features = extractor.extract(img_path)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"\nErro ao processar a imagem {img_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y

def build_manifest(metadata_path: str, images_dir: str) -> pd.DataFrame:
    """Constrói um DataFrame mapeando cada imagem utilizável ao seu diagnóstico."""
    metadata_path = Path(metadata_path)
    images_dir = Path(images_dir)

    df_meta = pd.read_excel(metadata_path)
    df_meta.columns = df_meta.columns.str.strip()

    img_to_patient = dict(zip(df_meta["image"], df_meta["patient_id"]))

    records: list[dict] = []

    for folder_path in images_dir.iterdir():
        if not folder_path.is_dir() or folder_path.name not in DG_CLASS_NAMES:
            continue

        dg_code = DG_CLASS_NAMES[folder_path.name]
        if dg_code in DG_EXCLUDED:
            continue

        binary_label = 1 if dg_code in DG_POSITIVE else 0

        for img_path in folder_path.glob("*.jpg"):
            img_name = img_path.name
            patient_id = str(img_to_patient.get(img_name, img_name.split("_")[0]))

            # Criamos um exam_id simples, mas agora a avaliação principal será por imagem
            exam_id = f"{patient_id}_ex"

            records.append({
                "image_path": str(img_path),
                "image": img_name,
                "patient_id": patient_id,
                "exam_id": exam_id,
                "dg_code": dg_code,
                "binary_label": binary_label,
            })

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {images_dir}")

    return manifest.reset_index(drop=True)


def _severity_tier(peak_dg: int) -> int:
    """Agrupa a gravidade para garantir balanceamento nos splits."""
    if peak_dg in (3, 4, 5): return 2  # Grave
    if peak_dg in (1, 2): return 1  # Leve/Moderado
    return 0  # Fisiológico


def split_patients(
        manifest: pd.DataFrame,
        test_size: float = 0.30, # 0.50 (50/50), 0.40 (60/40) ou 0.60 (40/60)
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divisão de pacientes em Treino e Teste.
    Garante que o mesmo paciente não apareça no treino e no teste,
    e mantém a proporção de gravidade (tiers) em ambos.
    """
    # 1. Descobre o diagnóstico mais grave de cada paciente para estratificar
    peak_dg_series = manifest.groupby("patient_id")["dg_code"].max().rename("peak_dg")

    patients = manifest.drop_duplicates("patient_id")[["patient_id"]].reset_index(drop=True)
    patients = patients.merge(peak_dg_series.reset_index(), on="patient_id")
    patients["tier"] = patients["peak_dg"].map(_severity_tier)

    # 2. Separação: Treino vs Teste
    train_pts, test_pts = train_test_split(
        patients,
        test_size=test_size,
        stratify=patients["tier"],
        random_state=random_state
    )

    # 3. Reconstrói os DataFrames de imagens baseados nos IDs sorteados
    train_df = manifest[manifest["patient_id"].isin(train_pts["patient_id"])].copy()
    test_df = manifest[manifest["patient_id"].isin(test_pts["patient_id"])].copy()

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

def split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Resumo estatístico da divisão dos dados em Treino e Teste."""
    rows = []
    for name, df in [("train", train_df), ("test", test_df)]:
        rows.append({
            "split": name,
            "patients": df["patient_id"].nunique(),
            "images": len(df),
            "pos_images": (df["binary_label"] == 1).sum(),
            "neg_images": (df["binary_label"] == 0).sum(),
        })
    return pd.DataFrame(rows).set_index("split")


# ── PyTorch Dataset ──────────────────────────────────────────────────────
class ROPDataset(Dataset):
    """
    Dataset compatível com as transformações nativas do torchvision (PIL).
    Retorna a imagem processada, a label (0 ou 1) e o exam_id.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]

        # Formato esperado pelo torchvision (PIL Image)
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)  # Retorna Tensor (3, H, W)

        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return image, label, row["exam_id"]