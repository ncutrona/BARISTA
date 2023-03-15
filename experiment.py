import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ERNB
import preprocess
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'serif'

#Comment
penalties = [0.01, 0.03, 0.06, 0.09, 0.12]
def cross_val(X, y, learning_rate, penalty):
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
        ernb = ERNB.ERNB()
        ernb.fit(X_train, y_train, penalty=penalty, learning_rate = learning_rate)
        ernb.predict(X_test, y_test)
        cross_accuracies.append(ernb.accuracy)
        split += 1
        if(split == 5):
            differences = ernb.norm_differnces()
    return np.mean(cross_accuracies), differences


def experiment(X, y, learning_rate, penalties):
    accuracies = []
    differences = []
    for i in range(len(penalties)):
        accuracy, difference = cross_val(X, y, learning_rate, penalties[i])
        accuracies.append(accuracy)
        differences.append(difference)
    max_value = max(accuracies)
    index = accuracies.index(max_value)
    return penalties[index], max_value, differences[index]



#Loading The Data (Enter File Path)
breast_w = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/breast_w.csv')
breast_w = preprocess.Preprocess(breast_w, "Class", [])
X, y = breast_w.get_data()
best_parameter, experimental_accuracy, breast_w_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("breast-w 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)

statlog = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/statlog.csv')
statlog = preprocess.Preprocess(statlog, "Target", ['age', 'resting_blood_pressure', 'serum_chol', 'max_hr', 'oldpeak'])
X, y = statlog.get_data()
best_parameter, experimental_accuracy, statlog_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("statlog 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)

iris = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/iris.csv')
iris = preprocess.Preprocess(iris, "variety", ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
X, y = iris.get_data()
best_parameter, experimental_accuracy, iris_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("iris 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)

krkp = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/krkp.csv')
krkp = preprocess.Preprocess(krkp, "36", [])
X, y = krkp.get_data()
best_parameter, experimental_accuracy, krkp_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("krkp 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)

mushroom = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/mushrooms.csv')
mushroom = preprocess.Preprocess(mushroom, "class", [])
X, y = mushroom.get_data()
best_parameter, experimental_accuracy, mushroom_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("mushroom 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)

zoo = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/zoo.csv')
zoo = preprocess.Preprocess(zoo, "17", [])
X, y = zoo.get_data()
best_parameter, experimental_accuracy, zoo_differences = experiment(X,y, learning_rate = 0.1, penalties=penalties)
print("zoo 5FCV Accuracy:", experimental_accuracy, "Best Penalty:", best_parameter)



#Convergence Plots
fig, ax = plt.subplots(3, 2, sharex=False, sharey=False, constrained_layout=True, figsize=[7, 7])
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('Semilog Plots of Weight Convergence', fontsize=12, fontweight='bold')

# Plot the subplots
# Plot 1
ax[0,0].semilogy(breast_w_differences, 'b')
ax[0,0].set_title('breast-w', fontsize=10)

# Plot 2
ax[0,1].semilogy(statlog_differences, 'b')
ax[0,1].set_title('statlog', fontsize = 10)

# Plot 3
ax[1,0].semilogy(iris_differences, 'b')
ax[1,0].set_title('iris', fontsize=10)

# Plot 4
ax[1,1].semilogy(krkp_differences, 'b')
ax[1,1].set_title('kr-vs-kp', fontsize=10)

# Plot 5
ax[2,0].semilogy(mushroom_differences, 'b')
ax[2,0].set_title('mushroom', fontsize = 10)

# Plot 6
ax[2,1].semilogy(zoo_differences, 'b')
ax[2,1].set_title('zoo', fontsize = 10)

# Adding a plot in the figure which will encapsulate all the subplots with axis showing only
fig.add_subplot(1, 1, 1, frame_on=False)

# Hiding the axis ticks and tick labels of the bigger plot
plt.tick_params(labelcolor="none", bottom=False, left=False)

# Adding the x-axis and y-axis labels for the bigger plot
plt.show()
