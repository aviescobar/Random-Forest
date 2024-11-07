from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
