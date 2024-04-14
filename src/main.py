"""
The lapidarist problem

aceron
"""

from scripts import eda, plots, models


def knn():
    """
    Main function
    """
    diamonds = eda.read_clean_diamonds()
    knn, y_predict, y_test, cols = models.knn_regression(diamonds)

    target = eda.pd.read_csv(eda.paths['missedDiamonds'])
    categorical_cols = ['cut', 'color', 'clarity']
    target = eda.load_and_encode(target, categorical_cols)

    predictions = knn.predict(target[cols])

    #print(f"Mean squared error: {error}")
    for idx, prediction in enumerate(predictions):
        print(f"Cost of diamond id: {idx} is $ {round(prediction, 2)}")    
    print("Acumulated cost: $", round(sum(predictions), 2))

    plots.plot_predictions(predictions)
    plots.plot_error_analysis(y_test, y_predict)

def linnear():
    """
    Main function
    """
    diamonds = eda.read_clean_diamonds()
    lr, y_predict, y_test = models.linear_regression(diamonds)

    target = eda.pd.read_csv(eda.paths['missedDiamonds'])
    predictions = lr.predict(target[['x','y','z','carat']])

    #print(f"Mean squared error: {error}")
    for idx, prediction in enumerate(predictions):
        print(f"Cost of diamond id: {idx} is $ {round(prediction[0], 2)}")    
    print("Acumulated cost: $", round(sum(predictions)[0], 2))

    plots.plot_predictions(predictions)
    plots.plot_error_analysis(y_test, y_predict)

def pca():
    diamonds = eda.read_clean_diamonds()
    pca_model = models.pca_analysis(diamonds)

    plots.plot_pca_results(pca_model, diamonds.columns)

def error_diff():
    diamonds = eda.read_clean_diamonds()
    
    _, y_predict, y_test = models.linear_regression(diamonds)
    lr_error = models.mean_squared_error(y_test, y_predict)

    _, y_predict, y_test, _ = models.knn_regression(diamonds)
    knn_error = models.mean_squared_error(y_test, y_predict)

    print(f"KNN error: {knn_error}")
    print(f"Linear Regression error: {lr_error}")
    print(f"Difference: {abs(knn_error - lr_error)}")


def price():
    diamonds = eda.read_clean_diamonds()
    plots.price_plot(diamonds)

if __name__ == "__main__":
    price()