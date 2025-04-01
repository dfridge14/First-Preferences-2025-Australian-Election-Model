setwd("/home/dania-freidgeim/Australian Election/")
Elec_similarity_df <- read.csv("Electorate_Similarity_for_R.csv", header = TRUE, stringsAsFactors = FALSE)
# 

similarity_model = lm(Response ~ Similarity + TCP_type + State, data = Elec_similarity_df)
summary(similarity_model)


# Load necessary libraries
library(ggplot2)
library(mgcv)  # For GAM models
library(quantreg)  # For quantile regression



# 1. Plot the raw data
ggplot(Elec_similarity_df, aes(x = Similarity, y = Response)) +
  geom_point(alpha = 0.2) + 
  geom_smooth(method = "lm", col = "red") + 
  ggtitle("Scatterplot with Linear Fit")

# 2. Linear Model
lm_model <- lm(Response ~ Similarity, data = Elec_similarity_df)
summary(lm_model)
plot(lm_model$residuals, main="Residuals of Linear Model")

# 3. Log Transformation of Predictor
Elec_similarity_df$LogSimilarity <- log(Elec_similarity_df$Similarity + 1)
lm_log <- lm(Response ~ LogSimilarity, data = Elec_similarity_df)
summary(lm_log)

ggplot(Elec_similarity_df, aes(x = LogSimilarity, y = Response)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", col = "blue") +
  ggtitle("Log-Transformed Similarity vs Response")

# 4. Polynomial Regression
lm_poly <- lm(Response ~ poly(Similarity, 2), data = Elec_similarity_df)
summary(lm_poly)

ggplot(Elec_similarity_df, aes(x = Similarity, y = Response)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), col = "green") +
  ggtitle("Polynomial Regression (Degree 2)")

# 5. Generalized Additive Model (GAM)
gam_model <- gam(Response ~ s(Similarity), data = Elec_similarity_df)
summary(gam_model)

ggplot(Elec_similarity_df, aes(x = Similarity, y = Response)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "gam", formula = y ~ s(x), col = "purple") +
  ggtitle("Generalized Additive Model (GAM)")

# 6. Quantile Regression (Median)
qr_model <- rq(Response ~ Similarity, tau = 0.5, data = Elec_similarity_df)
summary(qr_model)

ggplot(Elec_similarity_df, aes(x = Similarity, y = Response)) +
  geom_point(alpha = 0.2) +
  geom_abline(slope = coef(qr_model)[2], intercept = coef(qr_model)[1], col = "orange") +
  ggtitle("Quantile Regression (50th percentile)")



library(betareg)

# Fit Beta Regression
model_beta <- betareg(Response ~ Similarity + TCP_type + State - 1, 
                      data = Elec_similarity_df)

summary(model_beta)
plot(model_beta)

library(pscl)
pR2(model_beta)

ll_full <- logLik(model_beta)  # Log-likelihood of full model
ll_null <- logLik(betareg(Response ~ 1, data = Elec_similarity_df))  # Null model (intercept only)

pseudo_R2 <- 1 - (ll_full / ll_null)
pseudo_R2