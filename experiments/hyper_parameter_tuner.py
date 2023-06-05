import os
import random
import sys


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from keras_tuner import RandomSearch





from experiment_utils import Bunch, make_experiment, make_experiment_tempfile


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(Input(shape=(8,)))
    for i in range(hp.Int('num_layers', 5, 30)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=1024,
                                            step=32),
                               kernel_initializer='normal',
                        activation='relu'))
        #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(3, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [0.001])),
            loss='mean_absolute_error',
            metrics=['mean_absolute_error'])

    return model


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            split='fraction_of_coating',
            test_values=[40,50],
            epochs=1000,
            range_num_layers=[5,30],
            dense_units_range=[32,1024],
            activation=['relu'],
            kernel_initializer=['normal'],
            learning_rate=[0.001],
            optimizer='Adam',
            objective='val_mean_absolute_error',
            max_trials=15,
            executions_per_trial=1

            #range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        #Load dataset
        df = pd.read_excel('../data/database_new.xlsx')
        X = df.iloc[:, :8]
        Y = df.iloc[:, 25:28]
        """
        #Split dataset randomly into train and test

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.25,
            random_state=42)
            
        


        
        #Split on fractal dimension
        train_set = df[(df['fractal_dimension'] < 2.1) | (df['fractal_dimension'] > 2.2)]
        test_set = df[(df['fractal_dimension'] == 2.1) | (df['fractal_dimension'] == 2.2)]

        Y_train = train_set.iloc[:, 25:28]
        X_train = train_set.iloc[:, :8]
        Y_test = test_set.iloc[:, 25:28]
        X_test = test_set.iloc[:, :8]
        """

        #split on fraction of coating

        train_set=df[(df['fraction_of_coating']<40) | (df['fraction_of_coating']>50)]
        test_set=df[(df['fraction_of_coating']==40) | (df['fraction_of_coating']==50)]

        Y_train = train_set.iloc[:,25:28]
        X_train = train_set.iloc[:,:8]
        Y_test = test_set.iloc[:,25:28]
        X_test = test_set.iloc[:,:8]

        # Normalizing data
        scaling_x = StandardScaler()
        X_train = scaling_x.fit_transform(X_train)
        X_test = scaling_x.transform(X_test)

        #Tuner
        tuner = RandomSearch(build_model,
                             objective='val_mean_absolute_error',
                             max_trials=15,
                             executions_per_trial=1,
                             directory='project1',
                             project_name='fraction'
                             )
        print(tuner.search_space_summary())





        #Search plus Early stopping

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
        callback_list = [es]

        #Start search
        tuner.search(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])
        summary = tuner.results_summary()
        print(summary)

        #Print top 2 models
        models = tuner.get_best_models(num_models=2)

        best_model_1 = models[0]
        best_model_2 = models[1]
        # Build the model.
        # Needed for `Sequential` without specified `input_shape`.
        best_model_1.build()
        print(best_model_1.summary())

        best_model_2.build()
        print(best_model_2.summary())





