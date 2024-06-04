import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BARISTA
import preprocess
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'serif'

def cross_val(training_samples, training_labels, scheme, learning_rate, convergence_constant, max_iterations, l1_penalty, l2_penalty):
    cross_accuracies = []
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    split = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :],X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        barista = BARISTA.BARISTA()
        barista.fit(training_samples, training_labels, scheme, learning_rate, convergence_constant, max_iterations, l1_penalty, l2_penalty)
        barista.predict(X_test, y_test)
        cross_accuracies.append(barista.accuracy)
        split += 1
        if(split == 1):
            differences = barista.norm_differnces()
    return np.mean(cross_accuracies), differences


def experiment(training_samples, training_labels, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties):
    accuracies = []
    differences = []
    penalty_combo = []
    for i in range(len(l1_penalties)):
        for j in range(len(l2_penalties)):
            accuracy, difference = cross_val(training_samples, training_labels, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties[i], l2_penalties[j])
            accuracies.append(accuracy)
            differences.append(difference)
            penalty_combo.append([l1_penalties[i], l2_penalties[j]])
    max_value = max(accuracies)
    indices = [index for index, value in enumerate(accuracies) if value == max_value]
    return penalty_combo[indices[-1]], max_value, differences[indices[-1]]



#NUMERICAL SETTINGS
scheme = 'FISTA'
learning_rate = 0.1
convergence_constant = 1e-6
max_iterations = 5000
l1_penalties = [0.01, 0.03, 0.06, 0.09, 0.12]
l2_penalties = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05]


segmentation = pd.read_csv('/Users/nicolascutrona/Desktop/segmentation.csv')
segmentation = preprocess.Preprocess(segmentation, "REGION-CENTROID-COL", list(segmentation.columns[1:]))
X, y = segmentation.get_data()

best_parameters, experimental_accuracy, segmentation_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("segmentation 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, segmentation_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
#print("segmentation 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

plt.semilogy(segmentation_fista_differences, 'g')
plt.semilogy(segmentation_ista_differences, 'b')
plt.title('segmentation', fontsize=10, fontweight='bold')
plt.xlabel('Iterations')
plt.ylabel(r'$||W^* - W^{k}||_F$')

