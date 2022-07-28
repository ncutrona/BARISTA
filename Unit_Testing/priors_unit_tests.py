import numpy as np
import pandas as pd
import priors

def test_empty_input():
    assert [] == Priors([]).prior_vector
   
def test_one_group_only():
    assert [0.5, 0.5] == Priors([1,0]).prior_vector
     
def test_multi_instances_group():
    assert [0.5, 0.5] == Priors([1,0,1,0]).prior_vector

def test_type_input_all_digits():
    assert [0.25, 0.25, 0.25, 0.25] == Priors([1,0,2,3]).prior_vector

def test_type_input_all_chars():
    assert [0.5,0.5] == Priors(['a','b']).prior_vector

def test_mixed_type_input_chars_digits():
    assert [0.5,0.5] == Priors([1,'b']).prior_vector

def test_strings_as_values():
    assert [0.5, 0.5] == Priors(["abc", "xyz"]).prior_vector


test_empty_input()
test_one_group_only()
test_multi_instances_group()
test_type_input_all_digits()
test_type_input_all_chars()
test_mixed_type_input_chars_digits()
test_strings_as_values()