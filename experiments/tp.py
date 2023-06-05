import os
import random
import sys

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

#from sklearn.svm import SVR
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

#from sklearn.linear_model import Ridge

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
#from sklearn.multioutput import MultiOutputRegressor
import math
#from yellowbrick.regressor import ResidualsPlot
#from sklearn.metrics import r2_score
import sklearn.gaussian_process as gp


from experiment_utils import Bunch, make_experiment, make_experiment_tempfile



if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            split='fractal_dimension',
            split_type='interpolating',
            split_lower=-1,
            split_upper=-1,
            epochs=1000,
            patience=100,
            hidden_layers=2,
            batch_size=32,
            hidden_units=512,
            kernel_initializer='he_normal',
            # n_hidden=8,
            # dense_units=[416, 288, 256,256, 192,448,288,128, 352,224],
            # kernel_initializer=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'],
            activation='relu',
            loss='mean_squared_error'
            # range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        # Load dataset
        df = pd.read_excel('../data/database_new.xlsx')
        Y = df.iloc[:, 25:28]
        X = df.iloc[:, :8]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.25,
            random_state=42)

        scaling_x = StandardScaler()
        scaling_y = StandardScaler()
        X_train = scaling_x.fit_transform(X_train)
        X_test = scaling_x.transform(X_test)
        Y_train = scaling_y.fit_transform(Y_train)

        regressor = KernelRidge(alpha=0.0001, gamma=0.5, kernel='rbf')

        # wrapper=MultiOutputRegressor(regressor)
        model = regressor.fit(X_train, Y_train)
        # wrapper.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        # Y_pred=wrapper.predict(X_test)

        Y_pred = scaling_y.inverse_transform(Y_pred)

        error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')

        # error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
        print('Mean absolute error on test set: ', error)
        # Running and logging model plus Early stopping



        # logging Y_test values
        Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])
        # Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['q_abs']:
            _run.log_scalar('Actual q_abs', i)
        for i in Y_test['q_sca']:
            _run.log_scalar('Actual q_sca', i)
        for i in Y_test['g']:
            _run.log_scalar('Actual g', i)
        # logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["q_abs", "q_sca", "g"])
        for i in Y_pred['q_abs']:
            _run.log_scalar('Predicted q_abs', i)
        for i in Y_pred['q_sca']:
            _run.log_scalar('Predicted q_sca', i)
        for i in Y_pred['g']:
            _run.log_scalar('Predicted g', i)
        # logging difference between the two
        Y_diff = Y_test - Y_pred
        for i in Y_diff['q_abs']:
            _run.log_scalar('Absolute error q_abs', i)
        for i in Y_diff['q_sca']:
            _run.log_scalar('Absolute error q_sca', i)
        for i in Y_diff['g']:
            _run.log_scalar('Absolute error g', i)

        error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')

        # error=error*100
        print('Mean absolute error on test set [q_abs, q_sca, g]:-  ', error)
        _run.info['error'] = error
