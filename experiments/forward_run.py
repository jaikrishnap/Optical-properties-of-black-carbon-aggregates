import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
import pickle



from experiment_utils import Bunch, make_experiment, make_experiment_tempfile


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            epochs=1000,
            patience=200,
            hidden_layers=2,
            batch_size=32,
            hidden_units=512,
            kernel_initializer='he_normal',
            activation='relu',
            loss='mean_squared_error'
            #range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        #Load dataset
        df = pd.read_excel('../data/database_new.xlsx')
        X = df.iloc[:, :8]
        Y = df.iloc[:, 25:28]


        # Normalizaing Min max
        scaling_x = MinMaxScaler()
        scaling_y = MinMaxScaler()
        X = scaling_x.fit_transform(X)
        Y = scaling_y.fit_transform(Y)
        scalerfile_x = 'scaler_x.sav'
        scalerfile_y = 'scaler_y.sav'
        pickle.dump(scaling_x, open(scalerfile_x, 'wb'))
        pickle.dump(scaling_y, open(scalerfile_y, 'wb'))

        #Build NN model

        #model = build_model()#params.actuvation
        model = Sequential()
        model.add(Input(shape=(8,)))
        for j in range(0, params.hidden_layers):
            model.add(Dense(params.hidden_units, kernel_initializer=params.kernel_initializer, activation='relu'))

        model.add(Dense(3, kernel_initializer='glorot_normal', activation='sigmoid'))

        #Compile model
        model.compile(loss=params.loss, optimizer='adam',
                      metrics=['mean_absolute_error'])

        print(model.summary())


        #Running and logging model plus Early stopping

        filepath = f"forward_run_{_run._id}/best_model_forward.hdf5"
        with make_experiment_tempfile('best_model_forward.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
            #print(model_file.name)
            checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

            # # patient early stopping
            es = EarlyStopping(monitor='val_loss', patience=200, verbose=1)

            #log_csv = CSVLogger('fractal_dimension_loss_logs.csv', separator=',', append=False)

            callback_list = [checkpoint, es]
            history = model.fit(X, Y, epochs=params.epochs, batch_size=params.batch_size, validation_split=0.2, callbacks=callback_list)

            # choose the best Weights for prediction

            #Save the model

            #Save metrics loss and val_loss
            #print(history.history.keys())
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = len(loss)
            print(epochs)
            for epoch in range(0, epochs):

                # Log scalar wil log a single number. The string is the metrics name
                _run.log_scalar('Training loss', loss[epoch])
                _run.log_scalar('Validation loss', val_loss[epoch])

            #Use best model to predict
            weights_file = f'forward_run_{_run._id}/best_model_forward.hdf5'  # choose the best checkpoint
            model.load_weights(model_file.name)  # load it
            model.compile(loss=params.loss, optimizer='adam', metrics=[params.loss])
        # Evaluate plus inverse transforms on same data
        Y_pred = model.predict(X)
        Y_test = scaling_y.inverse_transform(Y)
        Y_pred = scaling_y.inverse_transform(Y_pred)

        #logging Y_test values
        Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])
        #Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['q_abs']:
            _run.log_scalar('Actual q_abs', i)
        for i in Y_test['q_sca']:
            _run.log_scalar('Actual q_sca', i)
        for i in Y_test['g']:
            _run.log_scalar('Actual g', i)
        #logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["q_abs", "q_sca", "g"])
        #print(Y_pred)
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



        #error=error*100
        print('Mean absolute error on same set [q_abs, q_sca, g]:-  ', error)
        _run.info['error'] = error

