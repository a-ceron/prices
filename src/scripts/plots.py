import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(predictions):
    #sorted_predictions = sorted([predictions[idx][0] for idx in range(len(predictions))])
    sorted_predictions = sorted(predictions)

    plt.figure(figsize=(10, 6))


    color = plt.cm.get_cmap('Reds',len(sorted_predictions))
    for i in range(len(sorted_predictions)):
        plt.bar(i, sorted_predictions[i], color=color(i / len(sorted_predictions)), edgecolor='black')
    for idx in range(len(sorted_predictions)):
        plt.text(idx, sorted_predictions[idx], f'${sorted_predictions[idx]:.2f}', ha='center', va='bottom')


    plt.title('Predictions of Diamond Prices')
    plt.xlabel('Diamond')
    plt.ylabel('Price')
    plt.xticks([])
    plt.yticks([])
    plt.box(False)  # Remove the frame
    plt.show()


def plot_pca_results(pca_model, columns):
    # Scree Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, pca_model.n_components_ + 1), pca_model.explained_variance_ratio_, marker='o', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Cumulative Explained Variance Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, pca_model.n_components_ + 1), np.cumsum(pca_model.explained_variance_ratio_), marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

        # Get the loadings matrix
    loadings = pca_model.components_

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=columns)
    plt.title('PCA Loadings Heatmap')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    plt.show()


def plot_error_analysis(y_true, y_pred):
    # Residual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true - y_pred, color='skyblue', alpha=0.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()

    # Scatter Plot of True vs. Predicted Labels
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='skyblue', alpha=0.5)
    plt.plot([(y_true.min()), (y_true.max())], [(y_true.min()), (y_true.max())], linestyle='--', color='red')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('True vs. Predicted Labels')
    plt.grid(True)
    plt.show()

    # Distribution of Errors
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Errors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Errors')
    plt.grid(True)
    plt.show()

    # Error vs. Feature Value Plot (if applicable)
    # Replace 'feature' with the name of the feature you want to analyze
    # plt.figure(figsize=(8, 6))
    # plt.scatter(data['feature'], errors, color='skyblue', alpha=0.5)
   
def visualise(df, vmin, vmax):
    
    df_sorted = df.sort_values(by='price')
    x = df_sorted['longitude']
    y = df_sorted['latitude']
    c = df_sorted['price'] 

    plt.rcParams['figure.figsize'] = [5, 6]
    plt.rcParams['figure.dpi'] = 100 

    plt.scatter(x, y, s=0.01, c=c, cmap='plasma_r', 
                norm=colors.Normalize(vmin=vmin,vmax=vmax), alpha=0.8)
    plt.colorbar()
    plt.show()

def price_plot(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'], kde=True, color='skyblue')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of Diamond Prices')
    plt.tight_layout()
    plt.show()
