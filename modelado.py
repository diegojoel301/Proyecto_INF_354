from sklearn.svm import SVC
import pickle

# Cargar los datos transformados
with open('train_test_data.pkl', 'rb') as file:
    X_train_tfidf, X_test_tfidf, y_train, y_test = pickle.load(file)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Guardar el modelo entrenado
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)