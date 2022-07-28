import pandas as pd
import numpy as np
import posteriors

def test_posterior_distribution():
  weight_matrix = np.array([[1],[1]])
  likelihood_matrix = np.array([[0.5], [0.5]])
  priors = [0.5,0.5]
  assert posteriors.PosteriorDistribution(weight_matrix, likelihood_matrix, priors).posterior_distribution == [0.5, 0.5]

test_posterior_distribution()