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
