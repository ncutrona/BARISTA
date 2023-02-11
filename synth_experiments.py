import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rpnb
plt.rcParams['font.family'] = 'serif'


#Loading The Data (Enter File Path)
synth = pd.read_csv('/Users/nicolascutrona/Desktop/RPNB Data/synth.csv')
synth = synth.drop('Unnamed: 0', axis = 1)

#Experimentation Set Up
beta_1 = 0
beta_2 = 0
test_split = 0.25
convergence = 1e-7
max_iterations = 5000
k = 5
lambda_terms = [0.01, 0.03, 0.06, 0.09, 0.12]

#Discretization
synth_continuous = []
synth_target = 'y'

#experiments
synth_model = rpnb.RPNB(0.001, [False, 1], k, synth, synth_target, test_split, synth_continuous, max_iterations, convergence, lambda_terms, beta_1, beta_2)
synth_accuracy = synth_model.final_model().test_accuracy
print("Synth Accuracy:", synth_accuracy)






