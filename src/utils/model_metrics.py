import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
