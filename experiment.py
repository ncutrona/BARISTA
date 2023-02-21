import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ERNB
import preprocess
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'serif'


#Loading The Data (Enter File Path)
breast_w = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/breast_w.csv')
breat_w = preprocess.Preprocess(breast_w, "Class", [])
y = breast_w['Class']
X = breast_w.drop('Class', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

ernb = ERNB.ERNB()
ernb.fit(X_train,y_train, learning_rate = 0.001)
ernb.predict(X_test, y_test)
print("Accuracy:", ernb.accuracy)
print("Predictions:", ernb.predictions)
print("Optimal_Weight_Parameters:", ernb.optimal_weights)
print("Posterior_Distributions:", ernb.posterior_distribution)


 



