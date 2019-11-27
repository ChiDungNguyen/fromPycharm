CREDIT_PATH = "/home/khanhan/Desktop/DataMining/dmba/GermanCredit.csv"

import pandas as pd

df = pd.read_csv(CREDIT_PATH)

df["RESPONSE"] = 1 - df["RESPONSE"]

X = df.drop(["OBS#", "RESPONSE"], axis=1)
Y = df["RESPONSE"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)

# Check distributions:
sum(y_train == 1) / len(y_train)
sum(y_test == 1) / len(y_test)

defaultXGB = XGBClassifier()
defaultXGB.fit(X_train, y_train)

# Predict label:
pred_label = defaultXGB.predict(X_test)

# Recall score:
from sklearn.metrics import recall_score

recall_score(y_test, pred_label)

# Confusion Matrix (CM):
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_label)

# Load thêm 8 classifiers từ Scikit-learn:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

# Kích hoạt 10 mô hình ML:
ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
gbc = GradientBoostingClassifier()
svc = SVC(probability=True)
ext = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier()

# List các mô hình ML:
models = [ran, knn, log, gbc, svc, ext, ada, gnb, gpc, bag]

# List trống lưu trung bình ROC/AUC của 5 lần thử nghiệm:
scores = []

# Huấn luyện 10 mô hình ML và tính trung bình AUC:
from sklearn.model_selection import cross_val_score

for mod in models:
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring="accuracy", cv=5, verbose=0)
    scores.append(acc.mean())



results = pd.DataFrame({
    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression',
              'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost',
              'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
    'Recall': scores})

result_df = results.sort_values(by='Recall', ascending=False).reset_index(drop=True)


def metric_cv(metric_selected):

    models = [ran, knn, log, gbc, svc, ext, ada, gnb, gpc, bag]

    # List trống lưu trung bình ROC/AUC của 5 lần thử nghiệm:
    scores = []

    # Huấn luyện 10 mô hình ML và tính trung bình AUC:
    from sklearn.model_selection import cross_val_score

    for mod in models:
        mod.fit(X_train, y_train)
        acc = cross_val_score(mod, X_train, y_train, scoring=metric_selected, cv=5, verbose=0)
        scores.append(acc.mean())

    results = pd.DataFrame({
        'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression',
                  'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost',
                  'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
        metric_selected: scores})

    result_df = results.sort_values(by=metric_selected, ascending=False).reset_index(drop=True)

    return result_df

metric_cv(metric_selected="roc_auc")



my_metrics = ["recall", "accuracy"]

for metrics in my_metrics:
    metric_cv(metric_selected=metrics)


