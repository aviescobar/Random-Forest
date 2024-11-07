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

