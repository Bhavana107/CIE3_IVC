import pandas as pd
from data_preprocessing import preprocess
from model import split_data, train_random_forest, predict
from evaluation import evaluate
import eda

# Load dataset
df = pd.read_csv("paddydataset.csv")

# EDA
eda.plot_histograms(df)
eda.plot_correlation(df)

# Preprocessing
X, y = preprocess("paddydataset.csv")

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
model = train_random_forest(X_train, y_train)

# Predict
y_pred = predict(model, X_test)

# Evaluate
evaluate(y_test, y_pred)