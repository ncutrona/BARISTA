import numpy as np
import pandas as pd
import discretize

data = pd.DataFrame(columns = ['A', 'y'])
data['A'] = [1] * 15
data['y'] = [1,1,1,1,1,2,2,2,2,3,3,3,4,4,5]
attribute = 'A'
class_attribute = 'y'
MLDP_test_object = discretize.MDLP(data, attribute, class_attribute)

def test_entropy_even():
  assert MLDP_test_object.entropy(pd.Series([1,1,0,0])) == 1

def test_entropy_odd():
  assert MLDP_test_object.entropy(pd.Series([1,1,0,1])) == 0.8112781244591328

def test_class_entropy_even():
  assert MLDP_test_object.class_entropy(pd.Series([1,1,0,0]), pd.Series([1,1]), pd.Series([0,0])) == 0 

def test_class_entropy_odd():
  assert MLDP_test_object.class_entropy(pd.Series([1,1,0,1]), pd.Series([1,1]), pd.Series([0,1])) == 0.5

def test_gain_even():
  assert MLDP_test_object.gain(pd.Series([1,1,0,0]), pd.Series([1,1]), pd.Series([0,0])) == 1

def test_gain_odd():
  assert MLDP_test_object.gain(pd.Series([1,1,0,1]), pd.Series([1,1]), pd.Series([0,1])) == 0.8112781244591328 - 0.5 

def test_change_ats():
  assert MLDP_test_object.change_ats(pd.Series([1,1,0,0]), pd.Series([1,1]), pd.Series([0,0])) == 0.8073549220576042

def test_check_split_condition():
  assert MLDP_test_object.check_split_condition(pd.Series([1,1,0,0]), pd.Series([1,1]), pd.Series([0,0])) == True

def test_best_split():
  assert MLDP_test_object.find_best_split(pd.Series([1,1,0,0])) == 2

def test_discretize():
  assert list(MLDP_test_object.dataframe['A']) == ["cat_1", "cat_1", "cat_1", "cat_1", "cat_1", "cat_2", "cat_2", "cat_2", "cat_2", "cat_3", "cat_3", "cat_3", "cat_4", "cat_4", "cat_5"]


test_entropy_even()
test_entropy_odd()
test_class_entropy_even()
test_class_entropy_odd() 
test_gain_even()
test_gain_odd()
test_change_ats()
test_check_split_condition()
test_best_split()
test_discretize()