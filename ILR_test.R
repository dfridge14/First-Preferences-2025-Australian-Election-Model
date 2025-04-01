# Load the compositions package
library(compositions)

# Step 1: Define the original proportions (summing to 1)
x <- c(0.4, 0.35, 0.15, 0.1)

# Step 2: Apply the ILR transformation
ilr_transformed <- ilr(x)

# Step 3: Back-transform the ILR values
recovered_proportions <- ilrInv(ilr_transformed)

# Output the results
cat("Original proportions: ", x, "\n")
cat("ILR transformed values: ", ilr_transformed, "\n")
cat("Recovered proportions after back-transformation: ", recovered_proportions, "\n")
cat("Sum of recovered proportions: ", sum(recovered_proportions), "\n")


library(compositions)
library(MASS)  # For mvrnorm (multivariate normal simulation)

# Step 1: Define the original proportions (summing to 1)
x <- c(0.4, 0.35, 0.15, 0.1)

# Step 2: Simulate multinomial data (1000 samples, 4 parts)
num_samples <- 1000
num_simulations <- 500

# Generate 500 simulations of multinomial data
simulated_data <- replicate(num_simulations, rmultinom(1, num_samples, prob = x))
simulated_data_matrix <- matrix(simulated_data, nrow = length(x), ncol = num_simulations)


# Step 3: Compute covariance matrix in the original (probability) space
simulated_proportions <- t(simulated_data_matrix) / num_samples
cov_matrix_prob_space <- cov(simulated_proportions)

# Print the covariance matrix in the original space
cat("Covariance matrix in original space (probability space):\n")
print(cov_matrix_prob_space)

# Step 4: Apply the ILR transformation to the simulated data
ilr_transformed <- t(apply(simulated_proportions, 1, ilr))

# Step 5: Compute covariance matrix in ILR space
cov_matrix_ilr_space <- cov(ilr_transformed)

# Print the covariance matrix in ILR space
cat("\nCovariance matrix in ILR space (transformed):\n")
print(cov_matrix_ilr_space)

# Step 6: Simulate multivariate normal in ILR space using the covariance matrix from the ILR space
simulated_mvn_ilr <- mvrnorm(num_simulations, mu = colMeans(ilr_transformed), Sigma = cov_matrix_ilr_space)

# Step 7: Back-transform the simulated MVN samples from ILR space to original space
back_transformed_mvn <- t(apply(simulated_mvn_ilr, 1, ilrInv))

# Step 8: Check the covariance matrix in the original space of the back-transformed data
cov_matrix_back_transformed <- cov(back_transformed_mvn)

# Print the covariance matrix of the back-transformed data
cat("\nCovariance matrix after back-transforming (original space):\n")
print(cov_matrix_back_transformed)

# Step 9: Calculate the distortion (difference between original covariance and back-transformed covariance)
cat("\nDistortion (difference between original and back-transformed covariance):\n")
distortion <- cov_matrix_prob_space - cov_matrix_back_transformed
print(distortion)
