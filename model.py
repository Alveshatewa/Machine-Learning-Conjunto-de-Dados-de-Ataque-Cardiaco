import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Carregar o dataset
df = pd.read_csv("Medicaldataset.csv")

# Exibir informações básicas
print(" Cabeçalho do dataset:")
print(df.head())

print("\n Informações do dataset:")
print(df.info())

# Verificar e tratar valores ausentes
print("\n Valores ausentes:")
print(df.isnull().sum())

# Codificar variável alvo (Result)
df['Result'] = df['Result'].map({'positive': 1, 'negative': 0})

# Exibir a distribuição da variável alvo
plt.figure(figsize=(6, 4))
sns.countplot(x='Result', data=df)
plt.title('Distribuição dos Casos de Ataque Cardíaco')
plt.xlabel('Resultado (0 = Negativo, 1 = Positivo)')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()

# Separar recursos (X) e alvo (y)
X = df.drop('Result', axis=1)
y = df['Result']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar o modelo com KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Avaliar o modelo
y_pred = knn.predict(X_test_scaled)

print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão (visual)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão - KNN')
plt.show()
