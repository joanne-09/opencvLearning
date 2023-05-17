# load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# /*---------------*/
# setups
# /*---------------*/

# lead digit dataset
digits = datasets.load_digits()
target = digits.target
flatten_digits = digits.images.reshape((len(digits.images), -1))

# visualize some handwritten images in the dataset
# set the size, row, col of the table
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4))
for ax, image, label in zip(axes, digits.images, target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('%i' % label)
# used to show the image above 
plt.show()

# divide images into training and test set
# set the test size to 20% of the total dataset
X_train, X_test, y_train, y_test = train_test_split(flatten_digits, target, test_size=0.2)


# /*---------------*/
# handwritten classification with logistic regression
# /*---------------*/

# standardize the dataset to put all the features of the variables on the same scale
scaler = StandardScaler()
X_train_logistic = scaler.fit_transform(X_train)
X_test_logistic = scaler.transform(X_test)

# create the logistic regression and fit it
# use the l1 panalty
# since this is a multiclass problem, paremeter multe_class is set to multinomial
logit = LogisticRegression(C=0.01, penalty='l1', solver='saga', tol=0.1, multi_class='multinomial')
logit.fit(X_train_logistic, y_train)
# predict the results
y_pred_logistic = logit.predict(X_test_logistic)
print("Accuracy: "+str(logit.score(X_test_logistic, y_test)))

# plot out the confusion matrix
# confusion matrix is a table used to define the performance of a classification algorithm
## these numbers represent how many times a number is defined to
# each row of the matrix represents the instance in a predicted class
# each column represents the instances in an actual class
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_logistic, labels=label_names)
df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for logistic regression"
plt.title(title)
plt.show()
# we can see that '8' is harder for computer to define correctly
# since it obtains the least correction of only 12 times correct


# /*---------------*/
# hand-written classification with SVM model
# use kernel(linear, poly, rbf) to divide datas
# SVM linear model
# use a line to divide two kinds of data
# /*---------------*/

# create and fit the SVM model to classify hand-written nums
svm_classifier = svm.SVC(gamma='scale')
# train the model
svm_classifier.fit(X_train, y_train)
# use test data to predict the model
y_pred_svm = svm_classifier.predict(X_test)
# get accuracy for the model
# it is nearly perfect(0.9916666666667)
print("Accuracy: "+str(accuracy_score(y_test, y_pred_svm)))

# check the confusion matrix
# the result is nearly perfect
label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmx = confusion_matrix(y_test, y_pred_svm, labels=label_names)
df_cm = pd.DataFrame(cmx)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
title = "Confusion Matrix for SVM results"
plt.title(title)
plt.show()


# /*---------------*/
# comparing both SVM and logistic regression with K-Fold Cross Validation
# k-fold cross validation is used when tehre limited samples
# we will add l2 regularization to visualize how well they both do
# /*---------------*/

algorithm = []
algorithm.append(('SVM', svm_classifier))
algorithm.append(('Logistic_L1', logit))
algorithm.append(('Logistic_L2', LogisticRegression(C=0.01, penalty='l2', solver='saga', tol=0.1, multi_class='multinomial')))

results = []
names = []
y = digits.target
for name, algo in algorithm:
    k_fold = model_selection.KFold(n_splits=10, random_state=10, shuffle=True)
    if name == 'SVM':
        X = flatten_digits
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(flatten_digits)
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')
        
    results.append(cv_results)
    names.append(name)

# plot the result
# SVM performs better all the time even with k-fold
# and it is better than both logistic regressions on average
fig = plt.figure()
fig.suptitle('Compare Logistic and SVM results')
ax = fig.add_subplot()
plt.boxplot(results)
plt.ylabel('Accuracy')
ax.set_xticklabels(names)
plt.show()
