"""
main_knn.py - Pipeline completo de Treinamento e Avaliação usando KNN.
"""

import pickle
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from Config import ConfigClassicML
from Dataset import build_manifest_zhao, split_patients, split_summary, build_tabular_dataset
from FeatureExtractor import FeatureExtractor


def main():
    print("="*50)
    print(" Iniciando Pipeline de Classificação ROP (KNN) ")
    print("="*50)

    # 1. Carregar Configurações
    cfg = ConfigClassicML()
    print(f"[INFO] Configurações carregadas.")
    print(f"       Semente: {cfg.seed} | Test Size: {cfg.test_size}")
    print(f"       Diretório de Resultados: {cfg.results_dir}\n")

    # 2. Construção do Manifest
    print(f"[INFO] Lendo metadados e arquivos de {cfg.images_dir}...")
    try:
        manifest = build_manifest_zhao(cfg.metadata_path, cfg.images_dir)
    except Exception as e:
        print(f"[ERRO] Falha ao construir manifest: {e}")
        return

    # 3. Divisão de Treino/Teste (Nível Paciente)
    print("[INFO] Realizando o split de pacientes estratificado pela gravidade...")
    train_df, test_df = split_patients(
        manifest,
        test_size=cfg.test_size,
        random_state=cfg.seed
    )
    
    print("\n--- Resumo da Divisão ---")
    print(split_summary(train_df, test_df))
    print("-" * 25 + "\n")

    # 4. Extração de Características
    extractor = FeatureExtractor(apply_clahe=cfg.clahe)
    
    print("[INFO] Extraindo características do conjunto de TREINO...")
    X_train, y_train = build_tabular_dataset(train_df, extractor)
    
    print("\n[INFO] Extraindo características do conjunto de TESTE...")
    X_test, y_test = build_tabular_dataset(test_df, extractor)

    # 5. Pré-processamento: Escalonamento
    # Fundamental para KNN, pois equaliza as escalas de cor e textura.
    print("\n[INFO] Aplicando StandardScaler para normalizar as características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Inicialização e Treinamento do Modelo KNN
    print(f"[INFO] Treinando o modelo KNN...")
    print(f"       -> K Vizinhos: {cfg.knn_neighbors}")
    print(f"       -> Pesos: {cfg.knn_weights}")
    print(f"       -> Métrica de Distância: {cfg.knn_metric}")
    
    knn = KNeighborsClassifier(
        n_neighbors=cfg.knn_neighbors,
        weights=cfg.knn_weights,
        metric=cfg.knn_metric
    )
    knn.fit(X_train_scaled, y_train)

    # 7. Avaliação do Modelo no Conjunto de Teste
    print("\n[INFO] Avaliando o modelo...")
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=> Acurácia Global: {acc:.4f}")

    print("\n=> Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=["Negativo (0)", "Positivo (1)"]))

    print("=> Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Salvamento dos Modelos
    # Precisamos salvar tanto o classificador quanto o scaler para inferências futuras
    model_path = Path(cfg.results_dir) / "knn_model.pkl"
    scaler_path = Path(cfg.results_dir) / "scaler.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"\n[INFO] Arquivos salvos com sucesso em: {cfg.results_dir}")
    print("       - knn_model.pkl")
    print("       - scaler.pkl")
    print("="*50)


if __name__ == "__main__":
    main()