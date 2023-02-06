import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rpnb
plt.rcParams['font.family'] = 'serif'


#Loading The Data (Enter File Path)
breast_w = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/breast_w.csv')
iris = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/iris.csv')
krkp = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/krkp.csv')
mushrooms = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/mushrooms.csv')
statlog = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/statlog.csv')
zoo = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/zoo.csv')

#Experimentation Set Up
test_split = 0.25
convergence = 1e-7
max_iterations = 5000
k = 5
beta_1 = 0.85
beta_2 = 0.15
lambda_terms = [0.01, 0.03, 0.06, 0.09, 0.12]

#Discretization
breast_w_continuous = []
breast_w_target = 'Class'

iris_continuous = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
iris_target = 'variety'

krkp_continuous = []
krkp_target = '36'

mushrooms_continuous = []
mushrooms_target = 'class'

statlog_continuous = ['age', 'resting_blood_pressure', 'serum_chol', 'max_hr', 'oldpeak']
statlog_target = 'Target'

zoo_continuous = []
zoo_target = '17'

#experiments
breast_w_model = rpnb.RPNB(0.01, [True, None], k, breast_w, breast_w_target, test_split, breast_w_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
breast_w_accuracy = breast_w_model.ten_fold_simulation()


statlog_model = rpnb.RPNB(0.01, [True, None], k, statlog, statlog_target, test_split, statlog_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
statlog_accuracy = statlog_model.ten_fold_simulation()


iris_model = rpnb.RPNB(0.01, [True, None], k, iris, iris_target, test_split, iris_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
iris_accuracy = iris_model.ten_fold_simulation()


krkp_model = rpnb.RPNB(0.05, [True, None], k, krkp, krkp_target, test_split, krkp_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
krkp_accuracy = krkp_model.ten_fold_simulation()


mushrooms_model = rpnb.RPNB(0.01, [True, None], k, mushrooms, mushrooms_target, test_split, mushrooms_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
mushrooms_accuracy = mushrooms_model.ten_fold_simulation()


zoo_model = rpnb.RPNB(0.01, [True, None], k, zoo, zoo_target, test_split, zoo_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
zoo_accuracy = zoo_model.ten_fold_simulation()


print("breast_w_accuracy:", breast_w_accuracy)
print("statlog_accuracy:", statlog_accuracy)
print("iris_accuracy:", iris_accuracy)
print("krkp_accuracy:", krkp_accuracy)
print("mushrooms_accuracy:", mushrooms_accuracy)
print("zoo_accuarcy:", zoo_accuracy)


#CONVERGENCE ANALYSIS
breast_test_model = breast_w_model.final_model()
breast_w_model_diff = breast_w_model.norm_differnces(breast_test_model) 

statlog_test_model = statlog_model.final_model()
statlog_model_diff = statlog_model.norm_differnces(statlog_test_model)

iris_test_model = iris_model.final_model()
iris_model_diff = iris_model.norm_differnces(iris_test_model)

krkp_test_model = krkp_model.final_model()
krkp_model_diff = krkp_model.norm_differnces(krkp_test_model)  

mushrooms_test_model = mushrooms_model.final_model()
mushrooms_model_diff = mushrooms_model.norm_differnces(mushrooms_test_model) 

zoo_test_model = zoo_model.final_model()
zoo_model_diff = zoo_model.norm_differnces(zoo_test_model)


#Convergence Plots
fig, ax = plt.subplots(3, 2, sharex=False, sharey=False, constrained_layout=True, figsize=[7, 7])
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('Semilog Plots of Weight Convergence', fontsize=12, fontweight='bold')

# Plot the subplots
# Plot 1
ax[0,0].semilogy(breast_w_model_diff, 'b')
ax[0,0].set_title('breast-w', fontsize=10)

# Plot 2
ax[0,1].semilogy(statlog_model_diff, 'b')
ax[0,1].set_title('heart-statlog', fontsize = 10)

# Plot 3
ax[1,0].semilogy(iris_model_diff, 'b')
ax[1,0].set_title('iris', fontsize=10)

# Plot 4
ax[1,1].semilogy(krkp_model_diff, 'b')
ax[1,1].set_title('kr-vs-kp', fontsize=10)

# Plot 5
ax[2,0].semilogy(mushrooms_model_diff, 'b')
ax[2,0].set_title('mushroom', fontsize = 10)

# Plot 6
ax[2,1].semilogy(zoo_model_diff, 'b')
ax[2,1].set_title('zoo', fontsize = 10)

# Adding a plot in the figure which will encapsulate all the subplots with axis showing only
fig.add_subplot(1, 1, 1, frame_on=False)

# Hiding the axis ticks and tick labels of the bigger plot
plt.tick_params(labelcolor="none", bottom=False, left=False)

# Adding the x-axis and y-axis labels for the bigger plot
plt.show()


