import numpy
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import discretize

class Preprocessing:

    def __init__(self, dataframe, target_attribute, test_split, continuous_attributes):
        self.dataframe = dataframe
        self.target_attribute = target_attribute
        self.test_split = test_split
        self.continuous_attributes = continuous_attributes
        self.discretize()
        self.training_samples, self.training_labels, self.testing_samples, self.testing_labels = self.split_data()
    
    def split_data(self):
        
        self.dataframe = shuffle(self.dataframe)
        y = self.dataframe[self.target_attribute]
        X = self.dataframe.drop(self.target_attribute, axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        return X_train, y_train, X_test, y_test
    
    def discretize(self):
        if(len(self.continuous_attributes)!= 0):
            print("_Discretiziig Data_...")
            for i in range(len(self.continuous_attributes)):
                self.dataframe = MDLP(self.dataframe, self.continuous_attributes[i], self.target_attribute).dataframe
        else:
            return