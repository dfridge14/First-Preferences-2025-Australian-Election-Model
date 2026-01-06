setwd("/home/dania-freidgeim/Australian Election/")
COAL_df <- read.csv("COAL_NP_ratio_df.csv", header = TRUE, stringsAsFactors = FALSE)

boxplot(NP_ratio ~ Incumbent, data = COAL_df) # significant difference (LP always incumbent)

Ratio_model = lm(NP_ratio ~ Incumbent, data = COAL_df)

summary(Ratio_model)

# significant Incumbent effect - reduces ratio by 0.13292