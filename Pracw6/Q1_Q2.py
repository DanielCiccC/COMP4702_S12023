import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, f1_score, fbeta_score

# ======================================================================================================================
# QUESTION 1

# Note that the data files contain extra spaces at the end confusing the read parameters, as a workaround we limit the
# maximum number of rows (must be a list-like of all strings for the function to work)
max_cols = 22

train_data = pd.read_csv('ann-train.data', sep=' ', header=None, usecols=range(max_cols))
test_data = pd.read_csv('ann-test.data', sep=' ', header=None, usecols=range(max_cols))

# As training and hold out validation sets are already defined we will convert these into their respective medical
# indicator inputs and classification values
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Developing a generalised K-NN Model with a k value of 5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Resulting confusion matrix printed below note the formatting is left to right i.e.
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"{cm}\n")

# ======================================================================================================================
# QUESTION 2

# Converting values to binary using the {Normal, Abnormal} convention for classifications
# x values of 1 & 2 being {hyperthyroid, hypothyroid} respectively
y_train_binary = (y_train != 3).astype(int)
y_test_binary = (y_test != 3).astype(int)

# Creation of the logistic regression model using sklearn
# Provides a probability that a point is a certain classification (in our case normal or abnormal)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_binary)

# Testing of multiple threshold values
thresholds = [0.1, 0.15, 0.3, 0.5, 0.65, 0.8, 0.95]
f1_scores = []
accuracies = []
fb_scores = []

for threshold in thresholds:
    # Provides a custom classification mapping where the probability of the point being classed as Normal or Abnormal is
    # used in conjunction with the threshold value which is used as the basis of 'acceptance'
    # (for the most part a lowered threshold means a higher value of false positives in this scenario)
    y_pred_custom = (log_reg.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    cm_custom = confusion_matrix(y_test_binary, y_pred_custom)
    print(f"Threshold: {threshold}\nConfusion Matrix:\n{cm_custom}\n")

    f1 = f1_score(y_test_binary, y_pred_custom)
    accuracy = accuracy_score(y_test_binary, y_pred_custom)
    # Beta value of 2 used, can be altered pending recall importance
    fb = fbeta_score(y_test_binary, y_pred_custom, beta=2)
    f1_scores.append(f1)
    accuracies.append(accuracy)
    fb_scores.append(fb)

# Part B - Graphing the Precision-Recall & ROC Curves
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_prob)
fpr, tpr, thresholds_roc = roc_curve(y_test_binary, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Precision-Recall Graph
# Since the problem is imbalanced & asymmetric this graph for decision-making of the model may be considered more useful
# Most data points are normal
# A false negative is considered more severe than a false positive
# THE GRAPH SHOWS THAT A TRADE-OFF OF A HIGH RECALL (Low False Negatives) COMES AT THE EXPENSE OF LOW PRECISION
# (Many False Positives)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()

# ROC Graph
# This graph is less informative as the problem is not balanced
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# F1 & Accuracy & Fb
# A Threshold of 0.5 has a very similar predictor of always predicting normal which has an accuracy of 0.927 but a
# worse F1 score of 0
plt.figure()
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.plot(thresholds, accuracies, label='Accuracy')
plt.plot(thresholds, fb_scores, label='Fb Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('F1 Score, Accuracy and Fb Score vs. Threshold')
plt.legend()
plt.grid()
plt.show()
