#OPTIMIZATION Class

#Needed Libraries
import numpy as np
import pandas as pd
import posterior
import likelihood
import prior


class Optimization:

    def __init__(self, training_samples, training_labels, max_iter, convergence_constant, learning_rate, lambda_term):
        print("_Computing Model Parameters_...\n")
        self.convergence_constant = convergence_constant #convergence constant to customize how much optimization is performed
        self.max_iter = max_iter #Max Iterations for optimization
        self.training_samples = training_samples #Training Data
        self.training_labels = training_labels #Training Targets
        self.label_values = np.unique(self.training_labels)
        self.num_observations = len(self.training_labels) #Number of Samples
        self.num_classes = len(self.label_values)
        self.lambda_term = lambda_term
        self.priors = prior.Priors(self.training_labels).prior_vector #Prior Vector
        self.ground_truths = self.get_ground_truth() #Ground Truth Values
        self.weight_matrix = self.initialize_weight_matrix() #Weight Matrix (will be updated through learning)
        self.learning_rate = learning_rate #WILL NEED TO IMPLEMENT LINE SEARCH
        self.likelihood_storage = self.get_likelihood_storage()
        self.likelihood_matrix_cache = self.get_likelihood_cache() #Gets a list of likelihood matrices for each sample
        self.posterior_cache = self.get_posterior_cache() #Gets a list of posterior distributions for each sample
        self.M = self.model_learning()
    
    #Initialize Learning Parameters
    def initialize_weight_matrix(self):

        attribute_length = len(self.training_samples.columns)

        return np.ones(shape = (self.num_classes, attribute_length))

    #Getting Needed Computations
    def get_ground_truth(self):
        
        #Getting a list of values to represent the ground truth for a sample
        ground_truths = []
        for i in range(self.num_observations):
            truth_value = self.training_labels[i]
            ground_truth_i = []
            for j in range(self.num_classes):
                if(truth_value == self.label_values[j]):
                    ground_truth_i.append(1)
                else:
                    ground_truth_i.append(0)
            ground_truths.append(ground_truth_i)
        
        return ground_truths

    #Cache Computations    
    def get_likelihood_cache(self):
        
        
        return likelihood.LikelihoodMatrix(self.training_samples, self.training_labels).likelihood_matrices

    def get_likelihood_storage(self):

        return likelihood.LikelihoodMatrix(self.training_samples, self.training_labels).likelihood_cache
        

    #Getting the cache of observation posterior distributions
    def get_posterior_cache(self): #Will be called each time the weight matrix is updated.

        likelihoods = self.likelihood_matrix_cache
        priors = self.priors
        posterior_distributions = []
        for i in range(self.num_observations): #Creating a list of posteriors for each sample 
            post_object = posterior.PosteriorDistribution(self.weight_matrix, likelihoods[i], priors)
            posterior_i = post_object.posterior_distribution
            posterior_distributions.append(posterior_i)

        return posterior_distributions #Returns the posterior distribution, as well as the simple and complex distributions
    

    #Model Learning
    def compute_iteration_loss(self):
        
        #Creates the loss function for the regularized implementation
        scale = 1/2
        loss = 0
        for i in range(self.num_observations):
            loss_i = 0
            for j in range(self.num_classes):
                loss_i += np.square((self.ground_truths[i][j] - self.posterior_cache[i][j]))
            loss += loss_i
    
        return scale * loss


    #partial for a weight matrix value
    def delta_Wij(self, class_index, attribute_index):
        
        d_sum = 0
        for i in range(self.num_observations):
            ground_truth = self.ground_truths[i][class_index]
            phat_estimation = self.posterior_cache[i][class_index]
            log_likelihood = np.log(self.likelihood_matrix_cache[i][class_index][attribute_index])

            d_sum += ((ground_truth - phat_estimation) * (phat_estimation * (1-phat_estimation) * log_likelihood))
        
        return d_sum


    def prox_solution(self, update):

        #Soft Thresholding
        boundary = (self.learning_rate * self.lambda_term) / 2
        

        if((-1 * boundary) < update and update < boundary):
            return 0
        
        elif(update > boundary):
            update -= boundary
            return update
        
        elif(update < (-1*boundary)):
            update += boundary
            return update
        
        else:
            print("Proximal Descent Error")
            return

    #Checks if the loss has stopped updating significantly (hard coded the constant for now)
    def convergence_check(self, loss_old, loss_new):

        converged = False
        convergence_check = np.abs((loss_old - loss_new)/max(np.abs(loss_old), np.abs(loss_new), 1)) #Making absolute value so negative values dont converge
        
        if(convergence_check < self.convergence_constant):
            converged = True
            return converged
        else:
            return converged
    
    def gradient_norm(self, gradient_wij):

        g_n_wij = round(np.sum(gradient_wij), 6)

        return g_n_wij

    #Optimization function
    def model_learning(self):
        
        weight_matrix = np.copy(self.weight_matrix)

        loss_values = []
        iteration = 0
        num_c = self.num_classes
        num_atts = len(self.training_samples.columns)
        converged = False

        if(self.max_iter > 0):
            print("_Optimizing_...\n")
        while(converged != True and iteration < self.max_iter):

            weight_matrix_gradient = []

            #Updating Weight Matrix
            for i in range(num_c):
                for j in range(num_atts):
                    gradient_ij = self.delta_Wij(i,j)
                    update = weight_matrix[i][j] - (self.learning_rate * gradient_ij)
                    proximal_solution = self.prox_solution(update)
                    weight_matrix[i][j] = proximal_solution
                    weight_matrix_gradient.append(weight_matrix[i][j])

       
            #Updating Framework Parameters
            self.weight_matrix = weight_matrix
            self.posterior_cache = self.get_posterior_cache()

            #Getting Gradient Values
            gradient_ij_norm = self.gradient_norm(weight_matrix_gradient)

            loss = self.compute_iteration_loss()
            loss_values.append(loss)

            #checking for convergence
            if(iteration > 0):
                converged = self.convergence_check(loss_values[-2], loss_values[-1])
    
            #Learning Output
            print("Iteration:", iteration+1)
            print("Posterior Cache First Sample:", self.posterior_cache[0])
            print("Weight Matrix:", self.weight_matrix)

            
            print("Gradient Weight Matrix Norm:", gradient_ij_norm)
            print("Model Loss:", loss)
            print("Converged:", converged)
            print("=================================================================")
            
            iteration += 1
            
            
        self.loss = loss_values
        
        #returning M
        if(converged):    
            return [self.weight_matrix]
            print("_Optimization Successful_")
        
        else:
            return [self.weight_matrix]
            print("_Optimization Failed_")
            

'''#Test 7: Making sure likelihood_matrix_cache is correct for all samples
data = pd.DataFrame(columns = ['x1', 'x2', 'y'])
data['x1'] = [1,1,0]
data['x2'] = [1,1,0]
data['y'] = ["y", "n", "y"]
test = Optimization(data[['x1', 'x2']], data['y'], 1, 1e-5, -0.01, False, None)
ground_truth = np.array([np.array([[0.75, 0.75],
        [0.5 , 0.5 ]]),
 np.array([[0.75, 0.75],
        [0.5 , 0.5 ]]),
 np.array([[0.25, 0.25],
        [0.5 , 0.5 ]])])
np.testing.assert_array_equal(test.likelihood_matrix_cache, ground_truth)


#Test 8: Making sure the posterior distributions all sum to one for each instance

assert [np.round(np.sum(x),2) == 1 for x in test.posterior_cache] == [True, True, True]

#Test 9: Making sure ground truth list is correct corresponding to the priors
assert test.ground_truths == [[0,1], [1,0], [0,1]]

#Test 10: Making sure iteration loss computation is correct
#assert round(test.compute_iteration_loss(),3) == 0.528

#Test 11: Making sure convergence is computing properly
assert test.convergence_check(0.2, 0.19999) == False
assert test.convergence_check(0.2, 0.199999) == True


#NEED TO UNIT TEST UPDATE FUNCTIONS'''