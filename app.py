import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np

# Configuração da página
st.set_page_config(page_title="Análise de Ataques Cardíacos", layout="wide")

# Título principal
st.title(" Análise de Conjuntos de Dados de Ataques Cardíacos com Machine Learning usando o Algoritmo (KNN)")
st.markdown("Plataforma para análise e predição de Conjuntos de Dados de Ataques Cardíacos usando algoritmos de Machine Learning.")

# Carregamento e pré-processamento dos dados
@st.cache_data
def load_data():
    df = pd.read_csv("Medicaldataset.csv")
    df['Result'] = df['Result'].map({'positive': 1, 'negative': 0})
    return df

df = load_data()

# Função para treinar modelo
@st.cache_resource
def train_model():
    X = df.drop('Result', axis=1)
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    metrics = {
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"])
    }
    return knn, metrics

# Sidebar com botões
st.sidebar.title("🔍 Navegação")
action = st.sidebar.radio("Escolha uma acção:", [
    "📄 Exibir Dataset",
    "📊 Exibir Distribuição",
    "📈 Mostrar Resultados",
    "📊 Avaliar Modelo",
    "ℹ️ Sobre a Aplicação",
    "📬 Contacto"
])

# Ações do sidebar
if action == "📄 Exibir Dataset":
    st.subheader("📄 Dataset Completo")
    st.dataframe(df)
    st.write("**Dimensão:**", df.shape)
    st.write("**Valores Ausentes:**")
    st.write(df.isnull().sum())

elif action == "📊 Exibir Distribuição":
    st.subheader("📊 Distribuição dos Casos de Ataque Cardíaco")
    fig, ax = plt.subplots()
    sns.countplot(x='Result', data=df, ax=ax)
    ax.set_xlabel("Resultado (0 = Negativo, 1 = Positivo)")
    ax.set_ylabel("Quantidade")
    ax.set_title("Distribuição dos Casos")
    st.pyplot(fig)



elif action == "📈 Mostrar Resultados":
    st.subheader("📈 Relatório de Classificação")
    _, metrics = train_model()
    st.code(metrics["classification_report"])

elif action == "📊 Avaliar Modelo":
    st.subheader("📊 Avaliação do Modelo com Métricas de Desempenho")
    _, metrics = train_model()

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.bar(['Acurácia'], [metrics['accuracy']], color='green')
        ax1.set_ylim(0, 1)
        ax1.set_title("Acurácia do Modelo")
        st.pyplot(fig1)
        st.markdown("🔹 **Acurácia**: indica a proporção total de previsões corretas.")

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(['Precisão'], [metrics['precision']], color='blue')
        ax2.set_ylim(0, 1)
        ax2.set_title("Precisão do Modelo")
        st.pyplot(fig2)
        st.markdown("🔹 **Precisão**: quantos dos positivos preditos são realmente positivos.")

    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.bar(['Recall'], [metrics['recall']], color='orange')
        ax3.set_ylim(0, 1)
        ax3.set_title("Recall do Modelo")
        st.pyplot(fig3)
        st.markdown("🔹 **Recall**: quantos dos positivos reais foram corretamente identificados.")

    with col4:
        fig4, ax4 = plt.subplots()
        sns.heatmap(metrics["confusion"], annot=True, fmt="d", cmap="Blues", ax=ax4,
                    xticklabels=["Negativo", "Positivo"],
                    yticklabels=["Negativo", "Positivo"])
        ax4.set_title("Matriz de Confusão")
        ax4.set_xlabel("Previsto")
        ax4.set_ylabel("Real")
        st.pyplot(fig4)
        st.markdown("🔹 **Matriz de Confusão**: mostra os acertos e erros do modelo.")

elif action == "ℹ️ Sobre a Aplicação":
    st.subheader("ℹ️ Sobre")
    st.markdown("""
    O diagnóstico precoce de ataques **cardíacos** é fundamental para salvar vidas e direcionar rapidamente pacientes para tratamentos adequados. No entanto, muitas instituições médicas enfrentam limitações de tempo e recursos humanos para realizar análises minuciosas dos dados clínicos. Isso pode levar a diagnósticos tardios ou incorretos. Diante disso, surge a necessidade de uma solução automatizada, confiável e de fácil aplicação que auxilie os profissionais da saúde na detecção precoce de possíveis infartos.
    
    A solução proposta baseia-se na criação de um modelo de classificação supervisionada utilizando o algoritmo KNN (K-Nearest Neighbors), aplicado a um conjunto de dados estruturado com informações clínicas de pacientes. O objetivo é prever, com base nos atributos fornecidos como: **pressão arterial, idade, colesterol e entre outros**, se um paciente está propenso a sofrer um ataque cardíaco **(classe positiva)** ou não **(classe negativa)**. 
    
    Com base nisto, esta aplicação permite:

    - Visualizar o dataset
    - Analisar a distribuição dos dados
    - Executar o modelo KNN
    - Avaliar o desempenho com gráficos explicativos
    - Entrar em contacto com o autor
    """)
    
elif action == "📬 Contacto":
    st.subheader("📬 Contactos")
    st.markdown("""
     Disponível para contacto por meio das seguintes redes sociais: 
     
    - [LinkedIn](https://www.linkedin.com/in/alveshatewa)  
    - [WhatsApp](https://wa.me/+244946602461)
    - [GitHub](https://github.com/alveshatewa)
    
     Desenvolvido por **Alves Hatewa** -  Estudante do Curso de Ciências da Computação.
    """)
