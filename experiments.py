import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rpnb

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
max_iterations = 1
k = 5
beta_1 = 0.85
beta_2 = 0.15
lambda_terms = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

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
breast_w_model = rpnb.RPNB(k, breast_w, breast_w_target, test_split, breast_w_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
breast_w_accuracy = breast_w_model.ten_fold_simulation()
iris_model = rpnb.RPNB(k, iris, iris_target, test_split, iris_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
iris_accuracy = iris_model.ten_fold_simulation()
krkp_model = rpnb.RPNB(k, krkp, krkp_target, test_split, krkp_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
krkp_accuracy = krkp_model.ten_fold_simulation()
mushrooms_model = rpnb.RPNB(k, mushrooms, mushrooms_target, test_split, mushrooms_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
mushrooms_accuracy = mushrooms_model.ten_fold_simulation()
statlog_model = rpnb.RPNB(k, statlog, statlog_target, test_split, statlog_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
statlog_accuracy = statlog_model.ten_fold_simulation()
zoo_model = rpnb.RPNB(k, zoo, zoo_target, test_split, zoo_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
zoo_accuracy = zoo_model.ten_fold_simulation()

print("breast_w accuracy:", breast_w_accuracy)
print("iris_accuracy:", iris_accuracy)
print("krkp_accuracy:", krkp_accuracy)
print("mushrooms_accuracy:", mushrooms_accuracy)
print("statlog_accuracy:", statlog_accuracy)
print("zoo_accuarcy:", zoo_accuracy)


#CONVERGENCE ANALYSIS
breast_test_model = breast_w_model.final_model()
breast_w_model.plot_convergence(breast_test_model) 

iris_test_model = iris_model.final_model()
iris_model.plot_convergence(iris_test_model)

krkp_test_model = krkp_model.final_model()
krkp_model.plot_convergence(krkp_test_model)  

mushrooms_test_model = mushrooms_model.final_model()
mushrooms_model.plot_convergence(mushrooms_test_model) 

statlog_test_model = statlog_model.final_model()
statlog_model.plot_convergence(statlog_test_model)

zoo_test_model = zoo_model.final_model()
zoo_model.plot_convergence(zoo_test_model)