import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from Config import cfg
from Dataset import build_manifest_zhao, split_patients, ROPDataset
from Transforms import build_transforms
from View import plot_dg_distribution, plot_roc_confusion_matrix, plot_pr_metrics_bar,salvar_tabela_csv


def get_densenet_extractor():
    """
    Carrega a DenseNet121 pré-treinada e remove a camada de classificação final.
    O objetivo é usar a rede apenas como um extrator de vetores densos (embeddings).
    """
    # Carrega a rede 121 com os pesos do ImageNet
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Pega apenas a parte extratora (features) e ignora o classificador linear
    extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
        nn.Flatten()  # Achata o tensor (agora de 1024x1x1 para 1024)
    )
    return extractor


def extract_deep_features(dataloader, model, device):
    """Passa os dados pela CNN e retorna as matrizes X (features) e y (labels)"""
    model.eval()
    X_list = []
    y_list = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Extraindo Embeddings"):
            images = images.to(device)
            # A rede gera um vetor gigante de características para cada imagem
            embeddings = model(images)

            X_list.append(embeddings.cpu().numpy())
            y_list.append(labels.numpy())

    # Empilha tudo em uma tabela Numpy padrão do scikit-learn
    X = np.vstack(X_list)
    y = np.concatenate(y_list).astype(np.int32)
    return X, y


def main():
    print("=== Pipeline Híbrido: DenseNet Embeddings + Naive Bayes ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilizando processamento via: {device}")

    # 1. Carregamento e Divisão dos Dados
    print("\n[1/5] Lendo metadados e separando pacientes...")
    manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)
    train_df, test_df = split_patients(manifest, test_size=cfg.test_size)

    dist_path = Path(cfg.results_dir) / "distribuicao_dados_nb_deep.png"
    plot_dg_distribution(train_df, test_df, dist_path)

    # 2. Configurando Datasets e DataLoaders do PyTorch
    print("\n[2/5] Carregando a DenseNet e processando imagens...")
    transform = build_transforms(size=cfg.img_size, clahe=cfg.clahe)

    train_dataset = ROPDataset(train_df, transform=transform)
    test_dataset = ROPDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    extractor = get_densenet_extractor().to(device)

    print("\n--- Conjunto de Treino ---")
    X_train, y_train = extract_deep_features(train_loader, extractor, device)

    print("\n--- Conjunto de Teste ---")
    X_test, y_test = extract_deep_features(test_loader, extractor, device)

    print(f"\n[!] Dimensionalidade das Features: {X_train.shape[1]} atributos por imagem!")

    # 3. Pré-processamento: Padronização
    print("\n[3/5] Padronizando as características extraídas pela CNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Treinamento do Modelo Clássico
    print(f"\n[4/5] Treinando o modelo Gaussian Naive Bayes (var_smoothing={cfg.nb_var_smoothing})...")
    nb_model = GaussianNB(var_smoothing=cfg.nb_var_smoothing)
    nb_model.fit(X_train_scaled, y_train)

    # 5. Avaliação do Modelo
    print("\n[5/5] Avaliando o modelo no Conjunto de Teste...")
    y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]

    # limiar
    limiar_customizado = cfg.threshold
    y_pred = (y_proba >= limiar_customizado).astype(int)

    # Cálculos exatos das métricas
    auroc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 90)
    print("RELATÓRIO DE MÉTRICAS - NAIVE BAYES (DENSENET FEATURES)")
    print("=" * 90)
    print(f"Limiar (Threshold) Usado  : {limiar_customizado:.2f} ({limiar_customizado * 100:.0f}%)")
    print(f"Pré-processamento         : {'CLAHE Ativado' if cfg.clahe else 'Padrão'}")
    print("-" * 90)
    print(f"AUROC                     : {auroc:.4f}")
    print(f"Acurácia Global           : {acc:.4f}")
    print(f"F1-Score                  : {f1:.4f}")
    print(f"Sensibilidade (Recall)    : {sensibilidade:.4f}")
    print(f"Especificidade            : {especificidade:.4f}")
    print("-" * 90)
    print("Matriz de Confusão:")
    print(f" - TP (Verdadeiros Positivos) : {tp}")
    print(f" - TN (Verdadeiros Negativos) : {tn}")
    print(f" - FP (Falsos Positivos)      : {fp}")
    print(f" - FN (Falsos Negativos)      : {fn}")
    print("=" * 90)

    # Gerando as imagens do Dashboard
    path_roc = Path(cfg.results_dir) / "grafico_roc_cm_nb_deep.png"
    path_pr = Path(cfg.results_dir) / "grafico_pr_metrics_nb_deep.png"

    plot_roc_confusion_matrix(y_test, y_proba, y_pred, auroc, "Naive Bayes (CNN Features)", path_roc)
    plot_pr_metrics_bar(y_test, y_proba, acc, sensibilidade, especificidade, f1, "Naive Bayes", path_pr)
    salvar_tabela_csv("NB com dense", auroc, acc, sensibilidade, especificidade, f1, cfg.clahe,
                      limiar_customizado)


if __name__ == "__main__":
    main()