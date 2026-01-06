import pymc as pm
import numpy as np
print(np.__version__)

# Generate some synthetic data
data = np.random.normal(0, 1, size=100)

# Build a basic model
if __name__ == '__main__':
    with pm.Model() as model:
        # Define prior for the mean
        mu = pm.Normal("mu", mu=0, sigma=1)
        
        # Define likelihood for the data
        likelihood = pm.Normal("data", mu=mu, sigma=1, observed=data)
        
        # Sample from the posterior
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
        #trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Check the trace
    print(trace)
    print("success")