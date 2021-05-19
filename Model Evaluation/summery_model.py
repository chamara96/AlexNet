from numpy import load
import idx2numpy
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

test_predicted = load('test_predicted.npy')

labelfile = "../MNIST_data_unziped/t10k-labels.idx1-ubyte"
labelarray = idx2numpy.convert_from_file(labelfile)

y_true = np.reshape(labelarray, [10000])
y_pred = np.reshape(test_predicted, [10000])

# Confusion Matrix
print('Confusion Matrix', confusion_matrix(y_true, y_pred))
print("=======================================")
# Accuracy
print('Accuracy:', accuracy_score(y_true, y_pred))
print("=======================================")
# Recall
print('Recall:', recall_score(y_true, y_pred, average=None))
print("=======================================")
# Precision
print('Precision', precision_score(y_true, y_pred, average=None))
print("=======================================")
# F1 Score
print('F1 Score:', f1_score(y_true, y_pred, average=None))
print("=======================================")
# classification report
print(classification_report(y_true, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
