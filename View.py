import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
import csv


def plot_dg_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: Path):
    """Plota a distribuição de Normal vs ROP (0 vs 1) nos splits de Treino e Teste."""
    data = []
    for name, df in [("Train", train_df), ("Test", test_df)]:
        counts = df['binary_label'].value_counts()
        for label, count in counts.items():
            data.append({"Split": name, "Status": "ROP (1)" if label == 1 else "Normal (0)", "Count": count})

    df_plot = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_plot, x="Split", y="Count", hue="Status", palette="Set2")
    plt.title('Distribuição de Classes (Normal vs ROP)', fontsize=14)
    plt.ylabel('Quantidade de Imagens')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Distribuição de dados salva em: {save_path}")


def plot_roc_confusion_matrix(y_true, y_scores, y_pred, auroc, model_name: str, save_path: Path):
    """Gera o Gráfico 1: Curva ROC e Matriz de Confusão Lado a Lado"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # curva auroc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {auroc:.4f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
    axes[0].set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    axes[0].set_title(f'Curva ROC - {model_name}', fontsize=14, pad=15)
    axes[0].legend(loc="lower right", fontsize=12)

    # matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Normal (0)', 'ROP (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                annot_kws={"size": 16})
    axes[1].set_xlabel('Previsão do Modelo', fontsize=12)
    axes[1].set_ylabel('Diagnóstico Real', fontsize=12)
    axes[1].set_title('Matriz de Confusão', fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Gráfico ROC + Matriz de Confusão salvo em: {save_path}")


def plot_pr_metrics_bar(y_true, y_scores, acc, sens, espec, f1, model_name: str, save_path: Path):
    """Gera o Gráfico 2: Curva de Precisão-Recall e Gráfico de Barras Lado a Lado"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    axes[0].plot(recall, precision, color='purple', lw=2, label=f'AP = {avg_precision:.4f}')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Recall (Sensibilidade)', fontsize=12)
    axes[0].set_ylabel('Precisão', fontsize=12)
    axes[0].set_title(f'Curva de Precisão-Recall - {model_name}', fontsize=14, pad=15)
    axes[0].legend(loc="lower left", fontsize=12)

    # metricas
    metricas_nomes = ['Acurácia', 'Sensibilidade', 'Especificidade', 'F1-Score']
    metricas_valores = [acc, sens, espec, f1]

    sns.barplot(x=metricas_nomes, y=metricas_valores, ax=axes[1], palette="viridis")
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_title('Desempenho Geral', fontsize=14, pad=15)
    axes[1].set_ylabel('Pontuação (0 a 1)', fontsize=12)

    # Adicionando os valores em cima de cada barra
    for i, v in enumerate(metricas_valores):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Gráfico PR + Métricas salvo em: {save_path}")

def salvar_tabela_csv(modelo: str, auroc, acc, sensibilidade, especificidade, f1, clahe: bool, limiar: float):
    """
    Salva os resultados formatados para criar a tabela comparativa da apresentação.
    Linhas: Cenário (Modelo + Clahe + Limiar) | Colunas: Métricas
    """
    csv_path = Path("tabela_comparativa_apresentacao1.csv")
    arquivo_existe = csv_path.exists()

    condicao_clahe = "C/ CLAHE" if clahe else "S/ CLAHE"

    # Ex: "Naive Bayes Manual | C/ CLAHE | Limiar 0.50"
    nome_cenario = f"{modelo} | {condicao_clahe} | Limiar {limiar:.2f}"

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Se o arquivo não existir, cria o cabeçalho das métricas na Horizontal
        if not arquivo_existe:
            writer.writerow(["Cenário Testado", "AUROC", "Acurácia", "Sensibilidade", "Especificidade", "F1-Score"])

        # Escreve a linha do cenário atual
        writer.writerow(
            [nome_cenario, f"{auroc:.4f}", f"{acc:.4f}", f"{sensibilidade:.4f}", f"{especificidade:.4f}", f"{f1:.4f}"])

    print(f"[✓] Tabela atualizada com sucesso no arquivo: {csv_path.name}")