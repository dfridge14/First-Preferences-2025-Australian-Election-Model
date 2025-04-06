library(dplyr)
library(car)

setwd("/home/dania-freidgeim/Australian Election/")
State_similarity_df <- read.csv("State_similarity_df_continuous.csv", header = TRUE, stringsAsFactors = FALSE)
# 

similarity_model = lm(Correlation ~ Similarity + TPPSwing + Adjacency - 1, data = State_similarity_df)
summary(similarity_model)
anova(similarity_model)

anova_table <- anova(similarity_model)
(anova_table$`Sum Sq`)[1:3]/sum((anova_table$`Sum Sq`)[1:3])

model_order_2 = lm(Correlation ~ TPPSwing + Similarity + Adjacency - 1, data = State_similarity_df)
anova(model_order_2)


Anova(similarity_model, type = "III")
Anova(model_order_2, type = "III")

# we go with coefficients of anova without-intercept!
anova = Anova(similarity_model, type = "III")
(anova$`Sum Sq`)[1:3]/sum((anova$`Sum Sq`)[1:3])

# Similarity, TPP_swing, Adjacency - 0.3828606 0.5830899 0.0340495






averaged_df = State_similarity_df %>%
  group_by(State1,State2, Adjacency, Correlation) %>%
  summarise(across(where(is.numeric), mean), .groups = "drop")


similarity_model_averaged = lm(Correlation ~ Similarity + TPPSwing + Adjacency - 1, data = averaged_df)
summary(similarity_model_averaged)