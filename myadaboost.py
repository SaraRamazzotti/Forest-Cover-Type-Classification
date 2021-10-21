import numpy as np

from decisionstump import DecisionStump
from utils import *

class MyAdaBoost:

    def __init__(self, n_estimators, n_classes, save_statistics=False, verbose=False):
        self._n_estimators = n_estimators
        self._n_classes = n_classes
        self._weak_classifiers = []
        self._weight_according_to_pi = []
        self._save_statistics = save_statistics
        self._verbose = verbose
        self._error_according_to_p_i = []

    def fit(self, data):
        '''
            Train the model.
            Args:
                self MyAdaBoost.
                data the complete dataframe (X,Y).
            Returns:
                self MyAdaBoost trained.
        '''
        
        self._weak_classifiers = []
        self._weight_according_to_pi = []
        self._error_according_to_p_i = []

        m, _ = data.shape

        for Y in self._n_classes:
        
            dataset = data.copy()
            dataset.iloc[:, -1] = dataset.apply(lambda x: 1 if x[-1]==Y else -1,axis=1)

            weak_classifiers_y = []
            weight_according_to_pi_y = []
            error_according_to_p_i_y = []
            training_error_y = []
            
            added = False
            for i in range(self._n_estimators):

                if i==0:
                    p_i = np.ones(m)*(1/m) # initialize, all training examples have the same weight 1/m
                  
                if self._verbose: print(f"p_i {p_i} for {Y} at round {i}")

                h_i = DecisionStump(save_statistics=self._save_statistics) 

                h_i = h_i.fit(dataset, p_i)
                
                weak_classifiers_y.append(h_i)

                y_pred = h_i.predict(dataset)
                list_of_mistakes = compute_l_i_t(y_pred, dataset.iloc[:, -1])
          
                error_of_h_i = compute_error_of_h_i(p_i, list_of_mistakes)

                if self._save_statistics:
                    error_according_to_p_i_y.append(error_of_h_i)

                rounded = round(error_of_h_i,4)
                if self._verbose: print(f"Error of h_i {rounded} for {Y} at round {i}")
                if rounded in (0, 0.0001, 0.4999, 0.5, 0.5001, 0.9999, 1):
                    if rounded in (0, 0.0001):                    
                        self._weak_classifiers.append([h_i])
                        self._weight_according_to_pi.append([1])
                        if self._verbose: print(f"error in zero T{i} c{Y}")
                        added = True
                        break
                    elif rounded in (0.4999, 0.5, 0.5001):
                        if i==0:
                          self._weak_classifiers.append(weak_classifiers_y)
                          self._weight_according_to_pi.append(weight_according_to_pi_y)
                        else:
                          self._weak_classifiers.append(weak_classifiers_y[:i])
                          self._weight_according_to_pi.append(weight_according_to_pi_y[:i])
                        if self._verbose: print(f"error in half T{i} c{Y}")
                        added = True
                        break
                    else:
                        self._weak_classifiers.append([h_i])
                        self._weight_according_to_pi.append([-1])
                        if self._verbose: print(f"error in one T{i} c{Y}")
                        added = True
                        break

                w_i = compute_w(error_of_h_i)
                weight_according_to_pi_y.append(w_i)
                
                if i < self._n_estimators-1:
                    p_i = compute_p_i_plus_one(p_i, list_of_mistakes, w_i)
                
                if self._verbose: print(f"w_i {w_i} for {Y} at round {i}")
            
            if not added:
                self._weak_classifiers.append(weak_classifiers_y)
                self._weight_according_to_pi.append(weight_according_to_pi_y)
            
            if self._save_statistics:
                self._error_according_to_p_i.append(error_according_to_p_i_y)

        return self

    def predict(self, examples, T=None, take_single=False):
        '''
            It predicts the multiclass classification.
            Args:
                self MyAdaBoost.
                examples the examples for whom we want to predict the labels.
                T the number of rounds at which we want the prediction, if None T is taken the default.
                take_single flag for taking or not the single binary predictions.
            Returns:
                the predicted labels and the predicted for each class if needed.
        '''

        if T==None:
            T = self._n_estimators
        
        sum_w_h = lambda w, pred : np.sum(np.multiply(np.array([w]).T, np.array([a.predict(examples) for a in pred])), axis=0)
        
        binary_predictions = [sgn(sum_w_h(w[:T], pred[:T])) for w, pred in zip(self._weight_according_to_pi, self._weak_classifiers)]
        
        predictions = np.argmax(binary_predictions, axis=0)

        if take_single: 
            return predictions+1, binary_predictions
        else:
            return predictions+1, None
    
    def error(self, examples, T=None, take_single=False):
        '''
            With this method can be computed the training error/test error of the fitted multiclass classifiers;
            if needed it can return the errors of every binary class prediction. 
            Args:
                self MyAdaBoost.
                examples the examples for whom we want to compute the error.
                T the number of rounds at which we want the error, if None T is taken the default.
                take_single flag for taking or not the single binary predictions.
            Returns:
                the error of the multiclass classification and the errors for each class if needed.
        '''     
        predicted, singles = self.predict(examples, T, take_single)
        m, _ = examples.shape
        data = (examples.iloc[:, -1]).to_numpy()

        full_error = (np.count_nonzero(np.not_equal(predicted, data)))/m

        if take_single:
            single_errors = [(np.count_nonzero(np.not_equal(p, np.vectorize(lambda x: 1 if x==self._n_classes[i] else -1)(data) )))/m for i, p in enumerate(singles)]
            return full_error, single_errors
        else:
            return full_error, None
    
    def get_error_according_to_p_i(self):
        '''
            The matrix with the weighted error of each weak classifier used during the train phase, 
            one row for each class and one column for each round.
            Args:
                self MyAdaBoost.
            Returns:
                matrix of the errors of the weak classifiers.
        '''
        return self._error_according_to_p_i