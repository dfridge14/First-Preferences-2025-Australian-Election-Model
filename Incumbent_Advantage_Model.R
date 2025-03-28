df <- read.csv("Incumbent_House_Senate_Final5_for_R.csv", header = TRUE, stringsAsFactors = FALSE)

df = df[df$div_nm != 'Mayo19',]



df$Diff_Pct <- df$House_Pct - df$Senate_Pct
df$PartyCat <- as.factor(df$PartyCat)
df$Demographic <- as.factor(df$Demographic)
df$State <- as.factor(df$State)

df$PartyCat <- relevel(df$PartyCat, ref = "Other")
df$Demographic <- relevel(df$Demographic, ref = "Inner Metropolitan")
df$State <- relevel(df$State, ref = "NSW")

model <- lm(Diff_Pct ~ PartyCat+ elections_won * Demographic, data = df)

summary(model)

# I think that this is the model to be using!!!! lm(Diff_Pct ~ PartyCat + elections_won * Demographic, data = df)









# Now, for the non-incumbents!

df2 <- read.csv("Non-incumbent_HS_for_R.csv", header = TRUE, stringsAsFactors = FALSE)


df2$incumbent_party <- as.factor(df2$incumbent_party)
df2$Ideology <- as.factor(df2$Ideology)

df2$incumbent_party <- relevel(df2$incumbent_party, ref = "ALP")
df2$Ideology <- relevel(df2$Ideology, ref = "ALP")

df2$incumbent_party[df2$incumbent_party == 'Other']  <- "ALP"

model2 <- lm(Diff_Pct ~ incumbent_party * Ideology, data = df2)

summary(model2)

# I like this model - everything significant (especially since our data is correlated so
# effective ss is ~> 4 times smaller))






# Final4 - x = 4; more data points!

df3 <- read.csv("Incumbent_House_Senate_Final4_for_R.csv", header = TRUE, stringsAsFactors = FALSE)

df3 = df3[df3$div_nm != 'Mayo19',]



df3$Diff_Pct <- df3$House_Pct - df3$Senate_Pct
df3$PartyCat <- as.factor(df3$PartyCat)
df3$Demographic <- as.factor(df3$Demographic)
df3$State <- as.factor(df3$State)

df3$PartyCat <- relevel(df3$PartyCat, ref = "Other")
df3$Demographic <- relevel(df3$Demographic, ref = "Inner Metropolitan")
df3$State <- relevel(df3$State, ref = "NSW")

model3 <- lm(Diff_Pct ~ PartyCat+ elections_won * Demographic, data = df3)

summary(model3)







# Install and load the MASS package
install.packages("MASS")


# Perform stepwise model selection using AIC
stepwise_model <- stepAIC(model, direction = "both")
