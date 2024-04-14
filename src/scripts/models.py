"""
models for the lapidarist problem

aceron 
"""
import pickle, os

import sklearn.linear_model as lm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def knn_regression(data):

    target = data.pop('price')

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, data.columns)
        ])
    # Train k-NN Regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', KNeighborsRegressor())])

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Guardar modelo
    with open('src/results/saves/knn_regression.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model, y_pred, y_test, X_test.columns


def pca_analysis(data):
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(data)

    # Se extrae el modelo entrenado del pipeline
    return pca_pipe.named_steps['pca']


def linear_regression(data):
    """
    Linear regression model
    """
    target = data[['price']].copy()
    properties = data[['x','y','z','carat']].copy()

    true_properties = properties.isna().sum(axis=1) == 0
    properties = properties[true_properties]
    target = target[true_properties]

    X_train, X_test, y_train, y_test = train_test_split(
        properties, target, test_size=0.33, random_state=42)

    # Entrenamiento
    lr = lm.LinearRegression()
    lr.fit(X_train, y_train)

    # Predicción
    y_pred = lr.predict(X_test)
    #error = mean_squared_error(y_test, y_pred)

    # Guardar modelo
    with open('src/results/saves/linear_regression.pkl', 'wb') as file:
        pickle.dump(lr, file)
    return lr, y_pred, y_test
