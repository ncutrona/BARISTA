import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ERNB
import preprocess
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'serif'


#Loading The Data (Enter File Path)
breast_w = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/breast_w.csv')
breast_w = preprocess.Preprocess(breast_w, "Class", [])
X, y = breast_w.get_data()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#X_train = X_train.reset_index(drop=True)
#X_test = X_test.reset_index(drop=True)
#y_train = y_train.reset_index(drop=True)
#y_test = y_test.reset_index(drop=True)


penalties = [0.01, 0.03, 0.06, 0.09, 0.12]
def cross_val(X, y, learning_rate, penalty):
    cross_accuracies = []
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :],X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        ernb = ERNB.ERNB()
        ernb.fit(X_train, y_train, penalty=penalty, learning_rate = learning_rate, max_iterations = 5000)
        ernb.predict(X_test, y_test)
        cross_accuracies.append(ernb.accuracy)
    return np.mean(cross_accuracies)


def experiment(X, y, learning_rate, penalties):
    accuracies = []
    for i in range(len(penalties)):
        accuracies.append(cross_val(X, y, learning_rate, penalties[i]))
    max_value = max(accuracies)
    index = accuracies.index(max_value)
    return penalties[index]


experimental_accuracy = experiment(X,y, learning_rate = 0.1, penalties=penalties)



    

######################################################################################################

'''krkp = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/krkp.csv')
krkp = preprocess.Preprocess(krkp, "36", [])
X, y = krkp.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, penalty = 0.06)
ernb.predict(X_test, y_test)
print("KRKP Accuracy:", ernb.accuracy)
#print("Predictions:", ernb.predictions)
#print("Optimal_Weight_Parameters:", ernb.optimal_weights)
#print("Posterior_Distributions:", ernb.posterior_distribution)'''


######################################################################################################

'''statlog = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/statlog.csv')
statlog = preprocess.Preprocess(statlog, "Target", ['age', 'resting_blood_pressure', 'serum_chol', 'max_hr', 'oldpeak'])
X, y = statlog.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, penalty = 0.01, convergence_constant=1e-8, learning_rate = 0.1)
ernb.predict(X_test, y_test)
print("Statlog Accuracy:", ernb.accuracy)
#print("Predictions:", ernb.predictions)
#print("Optimal_Weight_Parameters:", ernb.optimal_weights)
#print("Posterior_Distributions:", ernb.posterior_distribution)'''
 

######################################################################################################

'''iris = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/iris.csv')
iris = preprocess.Preprocess(iris, "variety", ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
X, y = iris.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, penalty = 0.01, convergence_constant=1e-8, learning_rate = 0.1)
ernb.predict(X_test, y_test)
print("Iris Accuracy:", ernb.accuracy)
#print("Predictions:", ernb.predictions)
#print("Optimal_Weight_Parameters:", ernb.optimal_weights)
#print("Posterior_Distributions:", ernb.posterior_distribution)'''


######################################################################################################


'''zoo = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/zoo.csv')
zoo = preprocess.Preprocess(zoo, "17", [])
X, y = zoo.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, penalty = 0.01, convergence_constant=1e-8, learning_rate = 0.1)
ernb.predict(X_test, y_test)
print("Zoo Accuracy:", ernb.accuracy)
#print("Predictions:", ernb.predictions)
#print("Optimal_Weight_Parameters:", ernb.optimal_weights)
#print("Posterior_Distributions:", ernb.posterior_distribution)'''


######################################################################################################


'''mushrooms = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/mushrooms.csv')
mushrooms = preprocess.Preprocess(mushrooms, "class", [])
X, y = mushrooms.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, penalty = 0.01, convergence_constant=1e-8, learning_rate = 0.1)
ernb.predict(X_test, y_test)
print("Mushrooms Accuracy:", ernb.accuracy)
#print("Predictions:", ernb.predictions)
#print("Optimal_Weight_Parameters:", ernb.optimal_weights)
#print("Posterior_Distributions:", ernb.posterior_distribution)'''

