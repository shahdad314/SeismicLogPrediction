import numpy as np

def mape_calc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def error_calc(label1, label2):
    """
    label1 is the measured value
    label2 is the predicted value
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    # from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import r2_score
    from math import sqrt

    mse = mean_squared_error(label1, label2)
    rmse = sqrt(mse)
    mae = mean_absolute_error(label1, label2)
    mape = mape_calc(label1, label2)
    r2 = r2_score(label1, label2)
    # print ("mse  = ", "{:.4f}".format(mse),"\nrmse = ", "{:.4f}".format(rmse)\
    #       , "\nmae  = ", "{:.4f}".format(mae), "\nmape = ", "{:.4f}".format(mape),\
    #       "\nr2   = ", "{:.4f}".format(r2))
    return mse, rmse, mae, mape, r2