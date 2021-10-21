import pandas as pd
import numpy as np


class DecisionStump:
    def __init__(self, save_statistics=False, verbose=False):
        self._i = None
        self._t = None
        self._risk = None
        self._training_error = None
        self._save_statistics = save_statistics
        self._verbose = verbose
  
    def fit(self, dataset, weights):
        '''
            Train the model.
            Args:
                self DecisionStump.
                dataset the complete dataframe (X,Y).
                weights the weights of each example.
            Returns:
                self DecisionStump trained.
        '''
        F_star = np.Inf
        m, d = dataset.shape

        dataset = dataset.to_numpy()
        Y = dataset[:, -1]
        X = np.delete(dataset, -1, axis=1)

        for j in range(d-1):
  
            j_features = X[:, j]
            sort_id_x = np.argsort(j_features)

            j_features = j_features[sort_id_x]
            j_features = np.append(j_features, j_features[-1]+1)
            
            weights_ordered = weights[sort_id_x]
            
            Y_ordered = Y[sort_id_x]
            
            F = np.sum(weights_ordered[np.where(Y_ordered==1)])
            
            if F < F_star:
                F_star = F
                teta_star = j_features[0] - 1
                j_star = j
            if self._verbose: print(f"Start {j} F* {F_star}, teta* {teta_star}, j* {j_star}")
            
            for i in range(m):
                F = F - Y_ordered[i] * weights_ordered[i]
                if self._verbose: print(f"F {F}  for column {j} round {i} {j_features[i]} != {j_features[i+1]}")
                if F < F_star and (j_features[i] != j_features[i+1]):
                    F_star = F
                    teta_star = (j_features[i] + j_features[i+1])/2
                    j_star = j
                if self._verbose: print(f"c{j}r{i} F* {F_star}, teta* {teta_star}, j* {j_star}")
            

        self._i = j_star
        self._t = teta_star

        if(self._save_statistics):
            self._risk = F_star
            predicted = self.predict(dataset)
            self._training_error = 1 - (np.count_nonzero(np.equal(predicted,dataset[:, -1])))/m

        return self

    def predict(self, testset):
        '''
            The prediction.
            Args:
                self DecisionStump.
                testset the examples for whom we want to predict the labels.
            Returns:
                The predicted labels.
        '''
        split = np.vectorize(lambda x: 1 if x<=self._t else -1)

        if isinstance(testset, pd.DataFrame):
            testset = testset.to_numpy()

        return split(testset[: , self._i])

    def get_risk(self):
        '''
            The minimum weighted training error obtained during the traing.
            Args:
                self DecisionStump.
            Returns:
                The risk.
        '''
        return self._risk

    def get_training_error(self):
        '''
            The training error.
            Args:
                self DecisionStump.
            Returns:
                The training error.
        '''
        return self._training_error
    
    def get_feature(self):
        '''
            The feature used for the prediction.
            Args:
                self DecisionStump.
            Returns:
                The index of the feature.
        '''
        return self._i

    def get_threshold(self):
        '''
            The threshold used for the prediction.
            Args:
                self DecisionStump.
            Returns:
                The threshold.
        '''
        return self._t