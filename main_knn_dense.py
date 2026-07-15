import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from Config import cfg
from Dataset import build_manifest_zhao, split_patients, split_summary, ROPDataset
from Transforms import build_transforms

from View import plot_dg_distribution, plot_roc_confusion_matrix, plot_pr_metrics_bar


def get_densenet_extractor():
    """
    Carrega a DenseNet121 pré-treinada e remove a camada de classificação final.
    O objetivo é usar a rede apenas como um extrator de vetores densos (embeddings).
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
        nn.Flatten()                  # Achata o tensor (1024 atributos)
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
            embeddings = model(images)

            X_list.append(embeddings.cpu().numpy())
            y_list.append(labels.numpy())

    X = np.vstack(X_list)
    y = np.concatenate(y_list).astype(np.int32)
    return X, y


def main():
    print("=== Pipeline Híbrido: DenseNet Embeddings + KNN ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilizando processamento via: {device}")

    # 1. Carregamento e Divisão dos Dados
    print("\n[1/5] Lendo metadados e separando pacientes...")
    manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)
    train_df, test_df = split_patients(manifest, test_size=cfg.test_size)

    dist_path = Path(cfg.results_dir) / "distribuicao_dados_knn_deep.png"
    plot_dg_distribution(train_df, test_df, dist_path)

    # 2. Configurando Datasets e DataLoaders do PyTorch
    print("\n[2/5] Carregando a DenseNet121 e processando imagens...")
    transform = build_transforms(size=cfg.img_size, clahe=cfg.clahe)

    train_dataset = ROPDataset(train_df, transform=transform)
    test_dataset = ROPDataset(test_df, transform=transform)

    # num_workers=0 resolve o problema do Windows com o multiprocessing do CLAHE
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    extractor = get_densenet_extractor().to(device)

    print("\n--- Conjunto de Treino ---")
    X_train, y_train = extract_deep_features(train_loader, extractor, device)

    print("\n--- Conjunto de Teste ---")
    X_test, y_test = extract_deep_features(test_loader, extractor, device)

    print(f"\n[!] Dimensionalidade das Features: {X_train.shape[1]} atributos por imagem!")

    # 3. Pré-processamento: Padronização
    # Fundamental para o KNN não sofrer distorções no cálculo da distância euclidiana
    print("\n[3/5] Padronizando as 1024 características extraídas pela CNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Treinamento do Modelo com Busca de Hiperparâmetros (GridSearchCV)
    print("\n[4/5] Realizando Validação Cruzada para encontrar o melhor K...")

    # Parâmetros que serão testados automaticamente
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring="f1")
    grid.fit(X_train_scaled, y_train)

    best_knn = grid.best_estimator_
    print(f"      -> Melhor configuração encontrada: {grid.best_params_}")

    # 5. Avaliação do Modelo no Conjunto de Teste
    print("\n[5/5] Avaliando o modelo no Conjunto de Teste...")
    y_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

    # Limiar customizado: 0.50 é a regra da maioria para o KNN (ex: 3 de 5 vizinhos = 60%)
    limiar_customizado = 0.90
    y_pred = (y_proba >= limiar_customizado).astype(int)

    # Cálculos exatos das métricas
    auroc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 90)
    print("RELATÓRIO DE MÉTRICAS - KNN (DENSENET FEATURES)")
    print("=" * 90)
    print(f"Hiperparâmetros Ótimos    : K={grid.best_params_['n_neighbors']}, Peso={grid.best_params_['weights']}, Dist={grid.best_params_['metric']}")
    print(f"Limiar (Threshold) Usado  : {limiar_customizado:.2f} ({limiar_customizado * 100:.0f}%)")
    print(f"Pré-processamento         : {'CLAHE Ativado' if cfg.clahe else 'Padrão'} + StandardScaler")
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

    # Gerando os 2 arquivos de imagem com os Gráficos
    path_roc = Path(cfg.results_dir) / "grafico_roc_cm_knn_deep.png"
    path_pr = Path(cfg.results_dir) / "grafico_pr_metrics_knn_deep.png"

    plot_roc_confusion_matrix(y_test, y_proba, y_pred, auroc, "KNN (CNN Features)", path_roc)
    plot_pr_metrics_bar(y_test, y_proba, acc, sensibilidade, especificidade, f1, "KNN", path_pr)

if __name__ == "__main__":
    main()