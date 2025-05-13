# CONJUNTO DE DADOS DE ATAQUE CARDÍACO

Este projeto realiza a análise e classificação de um dataset médico sobre um Conjunto de Dados de Ataque Cardíaco utilizando o algoritmo K-Nearest Neighbors (KNN).
O objetivo é prever se um paciente está em risco de ataque cardíaco com base em variáveis fornecidas no dataset.

## Estrutura do Projeto

- **model.py**: Script principal que realiza o carregamento, análise, pré-processamento e classificação dos dados.
- **Medicaldataset.csv**: Arquivo CSV contendo os dados médicos utilizados no projeto.

## Funcionalidades

1. **Carregamento do Dataset**: O script lê os dados do arquivo `Medicaldataset.csv`.
2. **Análise Exploratória**:
   - Exibição do cabeçalho e informações básicas do dataset.
   - Verificação de valores ausentes.
3. **Pré-processamento**:
   - Codificação da variável alvo (`Result`) para valores binários (1 = positivo, 0 = negativo).
   - Normalização dos dados utilizando `StandardScaler`.
4. **Divisão dos Dados**:
   - Separação em conjuntos de treino e teste (70% treino, 30% teste).
5. **Treinamento do Modelo**:
   - Utilização do algoritmo K-Nearest Neighbors (KNN) com 5 vizinhos.
6. **Avaliação do Modelo**:
   - Geração de um relatório de classificação.
   - Exibição de uma matriz de confusão para análise visual.

## Dependências

Bibliotecas Necessárias:

- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Modo de instalação das depências:

```bash
pip install pandas seaborn matplotlib scikit-learn
