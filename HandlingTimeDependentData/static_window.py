#==============================================================================
# Imports
#==============================================================================

from sklearn.linear_model import Lasso               # The model we will use
from sklearn.model_selection import ParameterSampler # A sampler for hyperparam
from math import ceil                                # Round up numbers
from scipy.stats import uniform                      # A uniform distribution

import numpy as np
import pandas as pd


#==============================================================================
# Prepare data
#==============================================================================

# Load Data
data = pd.read_csv('sp500.csv',header=0, delimiter = ",")

# Create features and labels
X = data[['Date','Close']]
X.loc[:,'Date'] = pd.to_datetime(X['Date']) # Convert to time data

# Create label vector
y = X.loc[2:,'Close']

# Create shifted columns
X = X.assign(shift = X['Close'].shift())
X = X.assign(double_shift = X['shift'].shift())

# Remove the two first rows with NaN values
X = X.loc[2:,['Date', 'shift', 'double_shift']]

#==============================================================================
# Adding time dependent variables
#==============================================================================

def time_dep_var(X, window, func, include_cur_obs = True):
    '''
    A function that applies a function on a window of lagged obeservations.
    The function assumes that X is sorted by time, such that time
    increases with the row number!

    X: Pandas array of data
    window: Int describing how far to look into the past
    func: Function that is applied to the data from the window
    include_cur_obs: Boolean determines if the current observations is also used by func

    Returns numpy array of new feature
    '''
    new_var = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if include_cur_obs:
            new_var[i] = func(X.iloc[np.max([0,i - window]):(i+1),:].values)
        else:
            new_var[i] = func(X.iloc[np.max([0,i - window]):i,:].values)

    return new_var

# Set window size
w_size = 30

# Add rolling mean
X = X.assign(rolling_mean=time_dep_var(X.loc[:,['shift']], w_size, np.mean))

# Add rolling standard deviation
X = X.assign(rolling_std_dev=time_dep_var(X.loc[:,['shift']], w_size, np.std))

#==============================================================================
# Score matrix, hyperparameters and loss function
#==============================================================================

# Number of hyperparameters to test
n_loops = 10

# Define or load a loss function. Here Mean Squared Error is used.
loss_mse = lambda x, y: np.mean((x-y)**2)

#==============================================================================
# Define time horizon
#==============================================================================

def equi_dist_time_points(start, end, step_size):
    '''
    This is a generator function which yields equidistant time points used for the static window fitting procedure
    '''
    # Calculate number of steps
    n_steps = ceil((end-start)/step_size)
    
    n = 0
    while n < n_steps:
        if n == n_steps - 1:
            yield start + n * step_size, end
        else:
            yield start + n * step_size, start + (n+1) * step_size
        n += 1


# Define a time point that separates training and validation data
end_train = np.datetime64('2014-01-01')

# Define time point that separates validation and test data
end_validation = np.datetime64('2016-01-01')

# Define step size as a timedelta, this is what numpy requires for division with datetime64 objects:
step_size = np.timedelta64(30,'D') 

# Create generator
time_points = equi_dist_time_points(end_train, end_validation, step_size)

#==============================================================================
# Initialize parameter sampler and model
#==============================================================================

# Dict with a distribution over the regularization parameter
lasso_reg_param = {'alpha': uniform(0,0.5)}

# Define parameter sampler
par_samp = ParameterSampler(lasso_reg_param, n_loops) # 10 indicates number of samples to draw

# Initialize model
mod = Lasso()

#==============================================================================
# Define Moving Window
#==============================================================================
def static_window(X, y, features_subset, model, time_points, parameter_sampler, loss_function):
    '''
    Function used for static window estimation.
    X: Pandas dataframe of features
    y: Pandas dataframe of labels Nx1
    features_subset: A list of names of features used in the model
    model: A model object that implements the scikit learn BaseEstimator interface and the methods fit and predict
    time_points: A generator that returns two consecutive time points
    parameter_sampler: A scikit learn parameter_sampler
    '''
    # Matrix to store loss and hyperparameters. One column for loss one for the hyperparameter in Lasso.
    score_matrix = np.zeros((n_loops, 2))

    # Initialize time point between train and validation
    split_point = None
    
    # Loop over parameters
    for idx, p in enumerate(parameter_sampler):

        # Vector to store predictions
        pred = np.empty(X.shape[0])

        # Loop over time steps
        for idx_time, t_vec in enumerate(time_points):
            # The generator returns a tuple. Split it into train and vali time points.
            t_train = t_vec[0]
            t_vali = t_vec[1]
            
            if idx_time == 0:
                # If this is the initial training round, save the training time point.
                split_point = t_train
            
            # Create boolean vector, with true for all obs in current time step
            cur_time_step = X['Date'].values <= t_train

            # Create boolean vector, with true for all obs in next time step
            next_time_step = np.logical_and(X['Date'].values > t_train, X['Date'].values <= t_vali)

            # Prepare data for current time step
            X_cur = X.loc[cur_time_step, features_subset] #['shift','double_shift']]
            y_cur = y.loc[cur_time_step]

            # Turn into numpy objects
            X_cur = X_cur.values
            y_cur = y_cur.values

            # Set parameters of the model
            model.set_params(**p)

            # Fit model
            model.fit(X_cur, y_cur)

            # Predict
            X_pred = X.loc[next_time_step, features_subset].values 
            pred[next_time_step] = model.predict(X_pred)

        # Calculate loss and store in score_matrix together with the sampled hyperparameter
        # Find index of all observations from the validation data
        validation_idx = X['Date'].values > split_point
        score_matrix[idx, 0] = loss_function(pred[validation_idx], y[validation_idx])

        # Get name of hyperparameter
        param_name = list(parameter_sampler.param_distributions.keys())[0]

        # Save hyperparameter value to the score matrix
        score_matrix[idx, 1] = p[param_name]

    return score_matrix

#==============================================================================
# Run on validation
#==============================================================================

# Identify the parameter set with the lowest corresponding loss
score_matrix = static_window(X, y, ['shift','double_shift'], mod, time_points, par_samp, loss_mse)
row_of_best_hyp_param = np.argmin(score_matrix[:,0])
print(score_matrix[row_of_best_hyp_param,:])
    
#==============================================================================
# Run on test
#==============================================================================

# Define the last time point in the test set
end_test = np.datetime64(X['Date'].max().strftime("%Y-%m-%d"))

# Create new time generator
time_points_test = equi_dist_time_points(end_validation, end_test, step_size)

# Create parameter sampler with only one value
best_reg_param = {'alpha': [score_matrix[row_of_best_hyp_param, 1]]}

# Define parameter sampler
par_samp_test = ParameterSampler(best_reg_param, 1) # 1 indicates number of samples to draw

# Evaluate best model on test data
score_matrix_test = static_window(X, y, ['shift','double_shift'], mod, time_points_test, par_samp_test, loss_mse)

# Print out score
print(score_matrix_test[0,0])
