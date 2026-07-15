from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from Config import cfg
from Dataset import build_manifest_zhao, split_patients, build_tabular_dataset
from FeatureExtractor import FeatureExtractor
from View import plot_dg_distribution, plot_roc_confusion_matrix, plot_pr_metrics_bar,salvar_tabela_csv

def main():
    print("=== Pipeline de Aprendizado de Máquina Clássico: KNN ===\n")

    # Carregamento e Divisão dos Dados
    print("[1/5] Lendo metadados e separando pacientes...")
    manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)
    train_df, test_df = split_patients(manifest, test_size=cfg.test_size)

    dist_path = Path(cfg.results_dir) / "distribuicao_dados_knn.png"
    plot_dg_distribution(train_df, test_df, dist_path)

    # Extração de Características
    print("\n[2/5] Extraindo características (Cor e Textura GLCM)...")
    extractor = FeatureExtractor(apply_clahe=cfg.clahe)

    print("\n--- Conjunto de Treino ---")
    X_train, y_train = build_tabular_dataset(train_df, extractor)

    print("\n--- Conjunto de Teste ---")
    X_test, y_test = build_tabular_dataset(test_df, extractor)

    # Pré-processamento: Padronização (Obrigatório para KNN)
    # Atributos em escalas diferentes distorcem a distância euclidiana geométrica do KNN.
    print("\n[3/5] Padronizando as características (Média 0, Variância 1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do Modelo com Busca de Hiperparâmetros (GridSearchCV)
    print("\n[4/5] Realizando Validação Cruzada para encontrar o melhor K...")

    # Parâmetros que serão testados automaticamente
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

    # Otimizando para F1-Score (ideal para dados médicos desbalanceados)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring="f1")
    grid.fit(X_train_scaled, y_train)

    best_knn = grid.best_estimator_
    print(f"      -> Melhor configuração encontrada: {grid.best_params_}")

    # Avaliação do Modelo
    print("\n[5/5] Avaliando o modelo no Conjunto de Teste...")

    # O KNN também gera probabilidades (proporção de vizinhos de uma classe)[cite: 1]
    y_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

    # limiar
    limiar_customizado = cfg.threshold # Para comparação justa com NB, usamos o padrão

    # criação do y_pred manualmente
    y_pred = (y_proba >= limiar_customizado).astype(int)

    # calculos
    auroc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 90)
    print("RELATÓRIO DE MÉTRICAS - KNN (BASELINE)")
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

    path_roc = Path(cfg.results_dir) / "grafico_roc_cm_knn.png"
    path_pr = Path(cfg.results_dir) / "grafico_pr_metrics_knn.png"

    plot_roc_confusion_matrix(y_test, y_proba, y_pred, auroc, "KNN", path_roc)
    plot_pr_metrics_bar(y_test, y_proba, acc, sensibilidade, especificidade, f1, "KNN", path_pr)
    salvar_tabela_csv("KNN sem dense", auroc, acc, sensibilidade, especificidade, f1, cfg.clahe,
                      limiar_customizado)

if __name__ == "__main__":
    main()