from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestRegressor
