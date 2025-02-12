install.packages("gee")
library(gee)
install.packages("geepack")
library(geepack)
install.packages("irr")
install.packages("magrittr")
library(magittr)
install.packages("tidyr")  
library(tidyr)
library(dplyr)


df_2022 <- read.csv("Final_x_HS_df.csv", header = TRUE, stringsAsFactors = FALSE)

df_2022$div_nm <- as.factor(df_2022$div_nm)


df_2022$Diff_Pct <- df_2022$House_Pct - df_2022$Senate_Pct
df_2022$Quot_Pct <- df_2022$House_Pct/df_2022$Senate_Pct


# format logical columns
class(df_2022$is_incumbent)
df_2022$is_incumbent = as.logical(df_2022$is_incumbent)
df_2022$is_historic_incumbent = as.logical(df_2022$is_historic_incumbent)

df_2022$is_incumbent <- ifelse(df_2022$is_incumbent == TRUE, 1, 0)
df_2022$is_historic_incumbent <- ifelse(df_2022$is_historic_incumbent == TRUE, 1, 0)
df_2022$is_incumbent [is.na(df_2022$is_incumbent )] <- 0
df_2022$is_historic_incumbent[is.na(df_2022$is_historic_incumbent)] <- 0


str(df_2022)
summary(df_2022)

colSums(is.na(df_2022))

table(df_2022$div_nm)

hist(df_2022$Senate_Pct, main="Histogram", xlab="Senate_Pct")
hist(df_2022$House_Pct, main="Histogram", xlab="House_Pct")
hist(df_2022$Diff_Pct, main="Histogram", xlab="Diff_Pct")


boxplot(df_2022$Quot_Pct ~ df_2022$is_incumbent, main="Boxplot by Group", 
        xlab="Group", ylab="Response Variable")

plot(df_2022$is_incumbent, df_2022$Diff_Pct, 
     main="Scatterplot", xlab="Predictor", ylab="Response")


# check correlation - reshape into wide for icc

df_2022_adj <- df_2022 |>
  group_by(div_nm) |>   # Ensure it's grouped by the cluster column
  mutate(orderid = rep(1:5, length.out = n())) |>
  ungroup()

df_wide <- df_2022_adj |> select(div_nm, orderid, Diff_Pct) |> pivot_wider(names_from = orderid, values_from = Diff_Pct)

library(irr)
icc(df_wide[, -1])

icc_result <- icc(df_2022[, c("Diff_Pct")], 
                  model = "oneway",  # You can use "twoway" depending on your data
                  type = "consistency",  # or "agreement" depending on your goal
                  unit = "single")
print(icc_result)



model <- geeglm(response ~ predictor1 + predictor2, 
                data = df_2022, 
                id = df_2022$div_nm, 
                family = gaussian(), 
                corstr = "exchangeable")


gee_model <- gee(Diff_Pct ~ is_incumbent + is_historic_incumbent, 
                 id = df_2022$div_nm, 
                 data = df_2022, 
                 family = gaussian, 
                 corstr = "exchangeable")