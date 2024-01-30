import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("mushrooms.csv", encoding = "ISO-8859-1")

# Separar la variable objetivo (class) y las características
X = data.drop('class', axis=1)
Y = data['class']

#Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (75% train, 25% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=11)

# Crear un modelo de árbol de clasificación
clf = DecisionTreeClassifier(random_state=11)

# Entrenar el modelo con el conjunto de entrenamiento
clf.fit(X_train, Y_train)

# Realizar predicciones en el conjunto de prueba
Y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Visualizar el árbol de clasificación si lo deseas
from sklearn.tree import export_text
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print(tree_rules)