"""
main_naive_bayes_dense.py - Pipeline de Naive Bayes usando características da DenseNet
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from Config import cfg
from Dataset import build_manifest_zhao, split_patients, split_summary, build_tabular_dataset

# 1. IMPORTAÇÃO ATUALIZADA: Usando o extrator da DenseNet
from FeatureExtractorDenseNet import DenseNetFeatureExtractor


def main():
    print("=== Pipeline Clássico: Naive Bayes c/ Features DenseNet ===\n")

    # 1. Carregamento e Divisão dos Dados
    print("[1/5] Lendo metadados e separando pacientes...")
    manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)

    train_df, test_df = split_patients(manifest, test_size=cfg.test_size)

    print("\nResumo da Divisão:")
    print(split_summary(train_df, test_df))

    # 2. Extração de Características
    # 2. ADAPTAÇÃO: Instanciando o extrator da DenseNet-121
    print("\n[2/5] Extraindo características profundas (DenseNet-121 - 1024 dimensões)...")
    extractor = DenseNetFeatureExtractor()

    print("\n--- Conjunto de Treino ---")
    X_train, y_train = build_tabular_dataset(train_df, extractor)

    print("\n--- Conjunto de Teste ---")
    X_test, y_test = build_tabular_dataset(test_df, extractor)

    # 3. Pré-processamento: Padronização (StandardScaler)
    # Embora o Naive Bayes seja menos sensível à escala,
    # faremos a padronização para manter a comparação com o KNN justa e exata.
    print("\n[3/5] Padronizando as características (Média 0, Variância 1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # IMPORTANTE: fit só no treino!

    # 4. Treinamento do Modelo
    print("\n[4/5] Treinando o modelo Gaussian Naive Bayes...")
    nb_model = GaussianNB(var_smoothing=cfg.nb_var_smoothing)
    nb_model.fit(X_train_scaled, y_train)

    # 5. Avaliação do Modelo
    print("\n[5/5] Avaliando o modelo no Conjunto de Teste...")
    y_pred = nb_model.predict(X_test_scaled)
    y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidade da classe positiva (doente)

    # --- Resultados ---
    print("\n" + "=" * 50)
    print("RESULTADOS FINAIS - NAIVE BAYES (DENSENET)")
    print("=" * 50)

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório de Classificação (Acurácia, Precisão, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=["Saudável (0)", "Doente (1)"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc:.4f}")

    # Bônus para a Apresentação/Arguição: Interpretabilidade
    print("\n[!] Dica para Arguição:")
    print("O Naive Bayes não é uma 'caixa preta'. Ele tomou essa decisão baseado nas seguintes médias aprendidas por classe:")
    print(f"Médias Classe Saudável (0): {nb_model.theta_[0][:4]}... (primeiros 4 de 1024 atributos)")
    print(f"Médias Classe Doente (1):   {nb_model.theta_[1][:4]}... (primeiros 4 de 1024 atributos)")


if __name__ == "__main__":
    main()