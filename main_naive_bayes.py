from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from Config import cfg
from Dataset import build_manifest_zhao, split_patients, build_tabular_dataset
from FeatureExtractor import FeatureExtractor
from View import plot_dg_distribution, plot_roc_confusion_matrix, plot_pr_metrics_bar,salvar_tabela_csv

def main():
    print("=== Pipeline de Aprendizado de Máquina: Naive Bayes ===\n")

    # Carregamento e Divisão dos Dados
    print("[1/5] Lendo metadados e separando pacientes...")
    manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)
    train_df, test_df = split_patients(manifest, test_size=cfg.test_size)

    dist_path = Path(cfg.results_dir) / "distribuicao_dados.png"
    plot_dg_distribution(train_df, test_df, dist_path)

    # Extração de Características
    print("\n[2/5] Extraindo características (Cor e Textura GLCM)...")
    extractor = FeatureExtractor(apply_clahe=cfg.clahe)

    print("\n--- Conjunto de Treino ---")
    X_train, y_train = build_tabular_dataset(train_df, extractor)

    print("\n--- Conjunto de Teste ---")
    X_test, y_test = build_tabular_dataset(test_df, extractor)

    # Pré-processamento: Padronização
    print("\n[3/5] Padronizando as características (Média 0, Variância 1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do Modelo
    print(f"\n[4/5] Treinando o modelo Gaussian Naive Bayes (var_smoothing={cfg.nb_var_smoothing})...")
    nb_model = GaussianNB(var_smoothing=cfg.nb_var_smoothing)
    nb_model.fit(X_train_scaled, y_train)

    # Avaliação do Modelo
    print("\n[5/5] Avaliando o modelo no Conjunto de Teste...")

    # extrai apenas as probabilidades contínuas da classe 1 (Doente)
    y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]

    # limiar
    limiar_customizado = cfg.threshold # 50 (padrao) ou 90 %

    # criação do y_pred manualmente
    # Se a probabilidade for maior ou igual ao limiar, vira 1. Se não, vira 0.
    y_pred = (y_proba >= limiar_customizado).astype(int)

    # calculos
    auroc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 90)
    print("RELATÓRIO DE MÉTRICAS - NAIVE BAYES")
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

    path_roc = Path(cfg.results_dir) / "grafico_roc_cm_nb.png"
    path_pr = Path(cfg.results_dir) / "grafico_pr_metrics_nb.png"

    plot_roc_confusion_matrix(y_test, y_proba, y_pred, auroc, "Naive Bayes", path_roc)
    plot_pr_metrics_bar(y_test, y_proba, acc, sensibilidade, especificidade, f1, "Naive Bayes", path_pr)
    salvar_tabela_csv("NB Sem dense", auroc, acc, sensibilidade, especificidade, f1, cfg.clahe,
                      limiar_customizado)

if __name__ == "__main__":
    main()