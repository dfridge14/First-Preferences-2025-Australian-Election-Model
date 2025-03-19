df <- read.csv("Incumbent_House_Senate_Final5_for_R.csv", header = TRUE, stringsAsFactors = FALSE)

df = df[df$div_nm != 'Mayo19',]

df$Diff_Pct <- df$House_Pct - df$Senate_Pct
df$PartyCat <- as.factor(df$PartyCat)
party_cat <- relevel(party_cat, ref = "Other")
model <- lm(Diff_Pct ~ PartyCat + elections_won, data = df)

summary(model)