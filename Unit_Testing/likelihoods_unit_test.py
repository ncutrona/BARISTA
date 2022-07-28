import numpy as np
import pandas as pd
import likelihoods

def test_likelihood_matrix():
  df = pd.DataFrame(columns = ['A', 'B'])
  df['A'] = [1,1,0,0]
  df['B'] = [1,0,1,0]
  assert np.array_equal(LikelihoodMatrix(df.drop('B', axis = 1), df['B']).likelihood_matrices, np.array([np.array([[0.5],[0.5]]), np.array([[0.5],[0.5]]), np.array([[0.5],[0.5]]), np.array([[0.5],[0.5]])])) == True

test_likelihood_matrix()