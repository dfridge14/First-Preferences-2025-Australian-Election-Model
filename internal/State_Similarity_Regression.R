

data <- read.csv("State_similarity_df_continuous.csv")

library(dplyr)

data <- data %>%
  filter(Election_Year != "All")

# Run linear regression
model <- lm(Correlation ~ Similarity + Adjacency + TPPSwing, data = data)

# Check results
summary(model)

library(ggcorrplot)  # For correlation heatmap

# Compute correlation matrix for all numerical columns
cor_matrix <- cor(df %>% select(where(is.numeric)), use = "complete.obs")

# Plot heatmap of correlations
ggcorrplot(cor_matrix, type = "lower", lab = TRUE)

library(lme4)
model2 <- lmer(Correlation ~ Similarity + Adjacency + TPPSwing + (Similarity | Election_Year), data = data)


agg_data <- data %>%
  group_by(State1, State2) %>%
  summarise(
    Correlation = first(Correlation),  # Since it's the same for all years
    Similarity = mean(Similarity),
    Adjacency = first(Adjacency),  # Since it's fixed
    TPPSwing = mean(TPPSwing)
  )

lm_model <- lm(Correlation ~ Similarity + Adjacency + TPPSwing, data = agg_data)
summary(lm_model)

