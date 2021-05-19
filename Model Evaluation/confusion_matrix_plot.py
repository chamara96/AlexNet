import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

from numpy import load

test_predicted = load('test_predicted.npy')
# print(data[0])

import idx2numpy

labelfile = "../MNIST_data_unziped/t10k-labels.idx1-ubyte"
labelarray = idx2numpy.convert_from_file(labelfile)

data = {'y_Actual': labelarray,
        'y_Predicted': test_predicted[0]
        }

df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt='d')       # fmt='.2%'     fmt="d"
plt.show()
