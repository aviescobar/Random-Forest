from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

# Datos de ejemplo (reemplaza esto con tus datos reales)
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
X_train_scaled = X_train * 0.5

# Entrenamiento de modelos
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

clf_rndr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
clf_rndr.fit(X_train, y_train_encoded)

clf_rndr_scaled = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
clf_rndr_scaled.fit(X_train_scaled, y_train_encoded)

# Función para evaluar resultados
def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    mse_without_scaling = metric(y_pred, y)
    mse_with_scaling = metric(y_prep_pred, y_prep)
    return mse_without_scaling, mse_with_scaling

# Ruta principal
@app.route('/')
def index():
    y_train_pred = clf_rndr.predict(X_train)
    y_train_prep_pred = clf_rndr_scaled.predict(X_train_scaled)

    mse_without_scaling, mse_with_scaling = evaluate_result(
        y_train_pred, y_train_encoded, y_train_prep_pred, y_train_encoded, mean_squared_error
    )

    
    y_train_pred, y_train_encoded, y_train_prep_pred, y_train_encoded, mean_squared_error
    )

 # Ruta para generar la gráfica
@app.route('/plot.png')
def plot_png():
    y_train_pred = clf_rndr.predict(X_train)
    y_train_prep_pred = clf_rndr_scaled.predict(X_train_scaled)

    plt.figure(figsize=(10, 5))

    # Gráfico de dispersión sin escalado
    plt.subplot(1, 2, 1)
    plt.scatter(y_train_encoded, y_train_pred, alpha=0.5, color='blue', label='Sin Escalar')
    plt.plot([y_train_encoded.min(), y_train_encoded.max()], [y_train_encoded.min(), y_train_encoded.max()], 'k--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Sin Escalar')
