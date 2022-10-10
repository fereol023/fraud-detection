# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:06:59 2022

@author: gbeno
"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from  sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

line = "============"*5
sep = "\n"

# lecture des données
dat = pd.read_csv(r"C:\Users\gbeno\Downloads\creditcard.csv")
print(dat.head())
print(dat.dtypes)
print(dat.isnull().sum())
print(line)

# Preprocessing
y = dat.iloc[:,-1]
X = dat.iloc[:,:-1]

print("Matrice des X")
print(X)
print(sep)
print(line)
print(y)

y = LabelBinarizer().fit_transform(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3)


# Estimation
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Evaluation
acc_score = accuracy_score(y_test, y_pred)
tpr = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
fpr = cm[0,1]/(cm[0,1]+cm[0,0])# spécificité
print(line)
print(cm)
print("Accuracy score : ",round(acc_score*100,2), " %.")
print("Recall score/taux de vraies fraudes détectées : ",round(tpr*100,2), " %.") 
print("Specificity score/fraudes non détectées : ",round(fpr*100,2), " %.")

print(line)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap="Blues")
plt.show()

# Validation croisée avec StratifiedKfold ou GridsearchCV ou Optuna
