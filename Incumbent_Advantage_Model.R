df <- read.csv("Incumbent_House_Senate_Final5_for_R.csv", header = TRUE, stringsAsFactors = FALSE)

df = df[df$div_nm != 'Mayo19',]



df$Diff_Pct <- df$House_Pct - df$Senate_Pct
df$PartyCat <- as.factor(df$PartyCat)
df$Demographic <- as.factor(df$Demographic)
df$State <- as.factor(df$State)

df$PartyCat <- relevel(df$PartyCat, ref = "Other")
df$Demographic <- relevel(df$Demographic, ref = "Inner Metropolitan")
df$State <- relevel(df$State, ref = "NSW")

model <- lm(Diff_Pct ~ PartyCat * elections_won, data = df)

summary(model)

# Install and load the MASS package
install.packages("MASS")
library(MASS)


# Perform stepwise model selection using AIC
stepwise_model <- stepAIC(model, direction = "both")
