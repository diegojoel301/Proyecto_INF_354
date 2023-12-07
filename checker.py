import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import sys
# Cargar el modelo y el vectorizador
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Cargar los datos de prueba
with open('train_test_data.pkl', 'rb') as file:
    _, X_test_tfidf, _, y_test = pickle.load(file)

# Realizar predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test_tfidf)

# Calcular la precisiÃ³n y mostrar el informe de clasificacion
accuracy = accuracy_score(y_test, y_pred_svm)
#print(f'Accuracy: {accuracy}')
#print(classification_report(y_test, y_pred_svm))

# Obtener las distancias de los vectores de soporte
y_scores = svm_model.decision_function(X_test_tfidf)

# Calcular el umbral optimo
threshold = np.percentile(y_scores, 75)

# Aplicar el umbral para la clasificacion
y_pred_adjusted = (y_scores > threshold).astype(int)

# Evaluar nuevamente con el nuevo umbral
#print('Adjusted Classification Report:')
#print(classification_report(y_test, y_pred_adjusted))

# Funcion para predecir una nueva entrada con ajuste de umbral
def predict_sql_injection_adjusted(query, threshold, vectorizer, model):
    query_tfidf = vectorizer.transform([query])
    score = model.decision_function(query_tfidf)
    prediction = (score > threshold).astype(int)
    return "Inyeccion SQL" if prediction == 1 else "Consulta SQL legi­tima"

# Ejemplo de uso con umbral ajustado
input_query = sys.argv[1]  # Reemplaza esto con tu entrada
result_adjusted = predict_sql_injection_adjusted(input_query, threshold, vectorizer, svm_model)
print(f'Prediction for "{input_query}": {result_adjusted}')