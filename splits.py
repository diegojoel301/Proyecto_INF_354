import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar tus datos
df = pd.read_csv('dataset.csv')
data = df.values.tolist()

# Función para calcular la mediana de la confiabilidad con ajuste de umbral
def calcular_mediana_confianza_con_umbral(data, test_size, n_splits, umbral_percentil):
    accuracies = []
    for _ in range(n_splits):
        # Mezclar los datos aleatoriamente y dividirlos
        random.shuffle(data)
        train_data = data[:int(len(data) * (1 - test_size))]
        test_data = data[int(len(data) * (1 - test_size)):]

        # Convertir a DataFrame y separar características y etiquetas
        train_df = pd.DataFrame(train_data, columns=df.columns)
        test_df = pd.DataFrame(test_data, columns=df.columns)
        X_train, y_train = train_df.drop('Query', axis=1), train_df['Label']
        X_test, y_test = test_df.drop('Query', axis=1), test_df['Label']

        # Entrenar el modelo SVC
        model = SVC(kernel='linear', random_state=42, probability=True)
        model.fit(X_train, y_train)

        # Calcular el umbral y ajustar las predicciones
        y_scores = model.decision_function(X_test)
        threshold = np.percentile(y_scores, umbral_percentil)
        y_pred_adjusted = (y_scores > threshold).astype(int)

        # Calcular y almacenar la precisión
        accuracies.append(accuracy_score(y_test, y_pred_adjusted))
    #print(accuracies)
    return np.median(accuracies)

# Calcular la mediana de la confiabilidad para el split académico (80/20)
mediana_academico = calcular_mediana_confianza_con_umbral(data, test_size=0.2, n_splits=100, umbral_percentil=75)
print(f'Mediana de la confiabilidad para el split academico: {mediana_academico}')

# Calcular la mediana de la confiabilidad para el split de investigación (50/50)
mediana_investigacion = calcular_mediana_confianza_con_umbral(data, test_size=0.5, n_splits=100, umbral_percentil=75)
print(f'Mediana de la confiabilidad para el split de investigacion: {mediana_investigacion}')
