#!/bin/bash

# Remove o arquivo CSV antigo para gerar uma tabela limpa e nova
echo "Limpando resultados antigos..."
rm -f tabela_comparativa_apresentacao.csv

# Loop pelas condições de CLAHE (True = Com CLAHE, False = Sem CLAHE)
for CLAHE_VAL in true false; do

    # Loop pelos limiares (0.50 ideal para NB, 0.90 para comparar extremo)
    for THRESH_VAL in 0.50 0.90; do

        echo "======================================================================="
        echo " INICIANDO BATERIA: CLAHE=$CLAHE_VAL | THRESHOLD=$THRESH_VAL "
        echo "======================================================================="

        # Exporta as variáveis de ambiente para o Python (o Config.py vai ler isso!)
        export CLAHE=$CLAHE_VAL
        export THRESHOLD=$THRESH_VAL

        # 1. Executa Naive Bayes Clássico
        python main_naive_bayes.py

        # 2. Executa KNN Clássico
        python main_knn.py

        # 3. Executa Naive Bayes alimentado pela DenseNet
        python main_naive_bayes_dense.py

        # 4. Executa KNN alimentado pela DenseNet
        python main_knn_dense.py

    done
done

echo "======================================================================="
echo " TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO! "
echo " Verifique a tabela final em: tabela_comparativa_apresentacao.csv "
echo "======================================================================="