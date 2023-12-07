import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Cargar el dataset
df = pd.read_csv('dataset.csv')

# Separar las caracterÃ­sticas y el objetivo
X = df['Query']  # Asumiendo que 'Query' es la columna con las consultas SQL
y = df['Label']  # Asumiendo que 'Label' es la columna con las etiquetas de inyecciÃ³n (0 o 1)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar TF-IDF Vectorizer y transformar los datos
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Guardar los datos transformados y el vectorizador para su uso posterior
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('train_test_data.pkl', 'wb') as file:
    pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), file)