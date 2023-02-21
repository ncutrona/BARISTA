import numpy
import pandas
import discretize


class Preprocess:

    def __init__(self, dataframe, target_attribute, continuous_attributes):
        self.dataframe = dataframe
        self.target_attribute = target_attribute
        self.continuous_attributes = continuous_attributes
        self.discretize()
        self.make_str()
        self.samples, self.labels = self.get_data()
    

    def get_data(self):
        labels = self.dataframe[self.target_attribute]
        samples = self.dataframe.drop(self.target_attribute, axis = 1)
        return samples, labels

    def discretize(self):
        if(len(self.continuous_attributes)!= 0):
            print("_Discretiziig Data_...")
            for i in range(len(self.continuous_attributes)):
                self.dataframe = discretize.MDLP(self.dataframe, self.continuous_attributes[i], self.target_attribute).dataframe
        else:
            return

    def make_str(self):
        string_cols = self.dataframe.columns[self.dataframe.dtypes==object].tolist()
        cols = self.dataframe.columns
        for i in range(len(cols)):
            if(cols[i] not in string_cols):
                self.dataframe[cols[i]] = self.dataframe[cols[i]].astype(str)