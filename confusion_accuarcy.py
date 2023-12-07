import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar los datos de prueba y el modelo entrenado
with open('train_test_data.pkl', 'rb') as file:
    _, X_test_tfidf, _, y_test = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Realizar predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test_tfidf)

# Calcular la precisiÃ³n y mostrar el informe de clasificaciÃ³n
#accuracy = accuracy_score(y_test, y_pred_svm)
#print(f'Accuracy: {accuracy}')
#print('Classification Report:')
#print(classification_report(y_test, y_pred_svm))

# Obtener las distancias de los vectores de soporte
y_scores = svm_model.decision_function(X_test_tfidf)

# Calcular el umbral Ã³ptimo
threshold = np.percentile(y_scores, 75)

# Aplicar el umbral para la clasificaciÃ³n
y_pred_adjusted = (y_scores > threshold).astype(int)

# Calcular la matriz de confusiÃ³n con el umbral ajustado
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print('Matriz de confusion:')
print(conf_matrix_adjusted)

# Calcular la precisiÃ³n con el umbral ajustado
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
print(f'Confiabilidad Ajustada: {accuracy_adjusted}')
