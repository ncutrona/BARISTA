import numpy as np
import pandas as pd

class MDLP:

    def __init__(self, dataframe, attribute, class_attribute):
        self.dataframe = dataframe
        self.attribute = attribute
        self.class_attribute = class_attribute
        self.cuts = []
        self.discretize(self.attribute)

    def ki(self, si):
        return np.unique(si)

    def entropy(self, s):
        k = len(self.ki(s))
        class_values = self.ki(s)
        count = s.value_counts()
        ent = 0
        for i in range(k):
            prob = count[class_values[i]]/len(s)
            ent += (prob * np.log2(prob))
        return -1 * ent

    def class_entropy(self, s, s1, s2):
        return ((len(s1)/len(s))*self.entropy(s1)) + ((len(s2)/len(s))*self.entropy(s2))

    def gain(self, s, s1, s2):
        return self.entropy(s) - self.class_entropy(s, s1, s2)

    def change_ats(self, s, s1, s2):
        return np.log2((3**len(self.ki(s))) - 2) - ((len(self.ki(s)) * self.entropy(s)) - (len(self.ki(s1)) * self.entropy(s1)) - (len(self.ki(s2)) * self.entropy(s2)))

    def check_split_condition(self, s, s1 ,s2):
        check = (np.log2(len(s) - 1) / len(s)) + (self.change_ats(s, s1, s2)/len(s))
        if(self.gain(s, s1, s2) > check):
            return True
        else:
            return False

    def make_series(self, s):
        s_prime = []
        for item in s:
            s_prime.append(item)
        s_prime = pd.Series(s_prime)
        return s_prime

    def find_best_split(self, s):
        if(len(s) > 1):
            true_index = list(s.index)
            s_prime = self.make_series(s)
            ents = []
            index_store = []
            for i in range(len(s_prime)):
                if(i != 0 and s_prime[i] != s_prime[i-1]):
                    s1 = s_prime[:i]
                    s2 = s_prime[i:]
                    ents.append(self.class_entropy(s_prime, s1, s2))
                    index_store.append(i)
            if(ents != []):
                min_ent = min(ents)
                index_setter = index_store[ents.index(min_ent)]
                return true_index[index_setter]
            else:
                return None
        else:
            return None

    def binary_mdlp(self, s):
        t_cut = self.find_best_split(s)
        if(t_cut != None):
            index_list = list(s.index)
            s1 = s[:index_list.index(t_cut)]
            s2 = s[index_list.index(t_cut):]
            t_cut_condition = self.check_split_condition(s, s1, s2)
            if(t_cut_condition):
                return t_cut
            else:
                return False
        else:
            return False

    def recursive_cut(self, s):
        res = self.binary_mdlp(s)
        if(res != False):
            index_list = list(s.index)
            self.cuts.append(res)
            self.cuts.sort()  
            s1 = s[:index_list.index(res)]
            s2 = s[index_list.index(res):]
            self.recursive_cut(s1)
            self.recursive_cut(s2)
            
    def multi_mdlp(self, attribute):
        df = self.dataframe[[attribute, self.class_attribute]]
        df = df.sort_values(attribute)
        df = df.reset_index(drop = True)
        s = df[self.class_attribute]
        self.recursive_cut(s)

    def discretize_helper(self, att_list, attribute):
        for i in range(len(att_list)):
            for j in range(len(att_list[i])):
                att_list[i][j] = "cat_" + str(i+1)
        discretized_values = [val for att_list in att_list for val in att_list]
        self.dataframe[attribute] = discretized_values  

    def discretize(self, attribute):
        att = list(self.dataframe[attribute])
        self.multi_mdlp(attribute)
        if(self.cuts == []):
            self.dataframe[attribute] = ["cat_1"]*len(self.dataframe[attribute])
        else:
            att = [att[s:e] for s, e in zip([0]+self.cuts, self.cuts+[None])]
            self.discretize_helper(att, attribute)