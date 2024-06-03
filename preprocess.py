import numpy
import pandas
import discretize


class Preprocess:

    def __init__(self, dataframe, target_attribute, continuous_attributes):
        self.dataframe = dataframe
        self.target_attribute = target_attribute
        self.continuous_attributes = continuous_attributes
        self.impute_missing()
        self.discretize()
        self.make_str()
        self.samples, self.labels = self.get_data()
    

    def get_data(self):
        labels = self.dataframe[self.target_attribute]
        samples = self.dataframe.drop(self.target_attribute, axis = 1)
        return samples, labels

    def impute_missing(self):
        columns = self.dataframe.columns
        nan_result = list(self.dataframe.isnull().any())
        for i in range(len(columns)):
            if(nan_result[i] == True):
                if(columns[i] in self.continuous_attributes):
                    self.dataframe[columns[i]].fillna(self.dataframe[columns[i]].mean(), inplace=True)
                else:
                    self.dataframe[columns[i]].fillna(self.dataframe[columns[i]].mode()[0], inplace=True)
            else:
                continue
        return self.dataframe


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
