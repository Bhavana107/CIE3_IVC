from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model = RandomForestClassifier(n_estimators=100, random_state=42)
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)