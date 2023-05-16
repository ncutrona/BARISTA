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



#ISTA vs FISTA Experiments

#=========================================================================================================================================================

breast_w = pd.read_csv('/filepath/breast_w.csv')
breast_w = preprocess.Preprocess(breast_w, "Class", [])
X, y = breast_w.get_data()

best_parameters, experimental_accuracy, breast_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("breast-w 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, breast_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("breast-w 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================

statlog = pd.read_csv('/filepath/statlog.csv')
statlog = preprocess.Preprocess(statlog, "Target", ['age', 'resting_blood_pressure', 'serum_chol', 'max_hr', 'oldpeak'])
X, y = statlog.get_data()

best_parameters, experimental_accuracy, statlog_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("statlog 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, statlog_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("statlog 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================

iris = pd.read_csv('/filepath/iris.csv')
iris = preprocess.Preprocess(iris, "variety", ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
X, y = iris.get_data()

best_parameters, experimental_accuracy, iris_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("iris 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, iris_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("iris 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================

krkp = pd.read_csv('/filepath/krkp.csv')
krkp = preprocess.Preprocess(krkp, "36", [])
X, y = krkp.get_data()

best_parameters, experimental_accuracy, krkp_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("krkp 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, krkp_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("krkp 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================

mushroom = pd.read_csv('/filepath/mushroom.csv')
mushroom = preprocess.Preprocess(mushroom, "class", [])
X, y = mushroom.get_data()

best_parameters, experimental_accuracy, mushroom_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("mushroom 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, mushroom_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("mushroom 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================

zoo = pd.read_csv('/filepath/zoo.csv')
zoo = preprocess.Preprocess(zoo, "17", [])
X, y = zoo.get_data()

best_parameters, experimental_accuracy, zoo_fista_differences = experiment(X, y, scheme, learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("zoo 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

best_parameters, experimental_accuracy, zoo_ista_differences = experiment(X, y, "ISTA", learning_rate, convergence_constant, max_iterations, l1_penalties, l2_penalties)
print("zoo 5FCV Accuracy:", experimental_accuracy, "Best Penalty Combination:", best_parameters)

#=========================================================================================================================================================



#Convergence Plots

fig, ax = plt.subplots(3, 2, sharex=False, sharey=False, constrained_layout=True, figsize=[7, 7])
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('Semilog Convergence', fontsize=12, fontweight='bold')

# Plot the subplots
# Plot 1
ax[0,0].semilogy(breast_fista_differences, 'g')
ax[0,0].semilogy(breast_ista_differences, 'b')
ax[0,0].set_title('breast-w', fontsize=10, fontweight='bold')
ax[0,0].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 2
ax[0,1].semilogy(statlog_fista_differences, 'g')
ax[0,1].semilogy(statlog_ista_differences, 'b')
ax[0,1].set_title('statlog', fontsize = 10, fontweight='bold')
ax[0,1].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 3
ax[1,0].semilogy(iris_fista_differences, 'g')
ax[1,0].semilogy(iris_ista_differences, 'b')
ax[1,0].set_title('iris', fontsize=10, fontweight='bold')
ax[1,0].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 4
ax[1,1].semilogy(krkp_fista_differences, 'g')
ax[1,1].semilogy(krkp_ista_differences, 'b')
ax[1,1].set_title('kr-vs-kp', fontsize=10, fontweight='bold')
ax[1,1].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 5
ax[2,0].semilogy(mushroom_fista_differences, 'g')
ax[2,0].semilogy(mushroom_ista_differences, 'b')
ax[2,0].set_title('mushroom', fontsize = 10, fontweight='bold')
ax[2,0].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 6
ax[2,1].semilogy(zoo_fista_differences, 'g')
ax[2,1].semilogy(zoo_ista_differences, 'b')
ax[2,1].set_title('zoo', fontsize = 10, fontweight='bold')
ax[2,1].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')



labels = ['FISTA', 'ISTA']
l1 = ax[0,0].semilogy(breast_fista_differences, 'g')
l2 = ax[0,0].semilogy(breast_ista_differences, 'b')
fig.legend([l1, l2], labels=labels,
           loc="upper right")

# Adding a plot in the figure which will encapsulate all the subplots with axis showing only
fig.add_subplot(1, 1, 1, frame_on=False)

# Hiding the axis ticks and tick labels of the bigger plot
plt.tick_params(labelcolor="none", bottom=False, left=False)

# Adding the x-axis and y-axis labels for the bigger plot
plt.savefig("/filepath/convergence.png", dpi=300)



fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, constrained_layout=True, figsize=[7, 3.5])
fig.suptitle('Semilog Convergence', fontsize=12, fontweight='bold')

# Plot the subplots
# Plot 1
ax[0].semilogy(breast_fista_differences, 'g')
ax[0].semilogy(breast_ista_differences, 'b')
ax[0].set_title('breast-w', fontsize=10, fontweight='bold')
ax[0].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')

# Plot 2
ax[1].semilogy(statlog_fista_differences, 'g')
ax[1].semilogy(statlog_ista_differences, 'b')
ax[1].set_title('statlog', fontsize = 10, fontweight='bold')
ax[1].set(xlabel = 'Iterations', ylabel = r'$||W^* - W^{k}||_F$')



labels = ['FISTA', 'ISTA']
l1 = ax[0].semilogy(breast_fista_differences, 'g')
l2 = ax[0].semilogy(breast_ista_differences, 'b')
fig.legend([l1, l2], labels=labels,
           loc="upper right")

fig.add_subplot(1, 1, 1, frame_on=False)

# Hiding the axis ticks and tick labels of the bigger plot
plt.tick_params(labelcolor="none", bottom=False, left=False)

# Adding the x-axis and y-axis labels for the bigger plot
plt.savefig("/filepath/convergencetwo.png", dpi=300)