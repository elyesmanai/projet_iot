from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from xgboost import XGBClassifier

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

from tabulate import tabulate

from tqdm import tqdm

import joblib

train = pd.read_csv('train_test_network.csv')
test = pd.read_csv('train_test_network.csv')

X_train = train.drop(['id','type','label'], axis=1)
y_train = train['label'] #0 is normal, 1 is attack

X_test = test.drop(['id','type','label'], axis=1)
y_test = test['label']

X_train.shape

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, objective='binary:logistic',tree_method='gpu_hist', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', random_state=42, probability=True, )
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression(random_state=42)
gnb = GaussianNB()

models = [('svm_proba', svm)]

results = []
for model_name, model in tqdm(models):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 3)
    precision = round(precision_score(y_test, preds), 3)
    recall = round(recall_score(y_test, preds), 3)
    f1 = round(f1_score(y_test, preds, average='weighted'), 3)
    results.append([model_name, accuracy, precision, recall, f1])
    
    joblib.dump(model, f'models/{model_name}.joblib')
    
table = tabulate(results, headers=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'], tablefmt='grid')
print(table)


for feature in X_train.columns:
    if X_train[feature].nunique() == 1:
        X_train.drop(feature, axis=1, inplace=True)
        X_test.drop(feature, axis=1, inplace=True)