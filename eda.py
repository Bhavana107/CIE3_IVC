import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df):
    df.hist(figsize=(10, 8))
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Matrix")
    plt.show()

def scatter_plot(df, col1, col2):
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("Scatter Plot")
    plt.show()