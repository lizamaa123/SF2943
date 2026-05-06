library(zoo)

# From STEP 1-4
df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)
log_returns <- diff(log(df$CBBTCUSD))

# ARMA(1, 1)
model_arma11 <- arima(log_returns, order = c(1, 0, 1))

# ARMA(1, 6)
model_arma16 <- arima(log_returns, order = c(1, 0, 6))

# ARMA(4, 1)
model_arma41 <- arima(log_returns, order = c(4, 0, 1))

# ARMA(4, 6)
model_arma46 <- arima(log_returns, order = c(4, 0, 6))

# AR(1)
model_ar1 <- arima(log_returns, order = c(1, 0, 0))

# MA(1)
model_ma1 <- arima(log_returns, order = c(0, 0, 1))

cat("\n--- Model Selection (AIC Scores) ---\n")
cat("ARMA(1,1) AIC: ", AIC(model_arma11), "\n")
cat("ARMA(1,6) AIC: ", AIC(model_arma16), "\n")
cat("ARMA(4,1) AIC: ", AIC(model_arma41), "\n")
cat("ARMA(4,6) AIC: ", AIC(model_arma46), "\n")
cat("AR(1) AIC:     ", AIC(model_ar1), "\n")
cat("MA(1) AIC:     ", AIC(model_ma1), "\n")

scores = c(AIC(model_arma11), AIC(model_arma16), AIC(model_arma41), AIC(model_arma46), AIC(model_ar1), AIC(model_ma1))
model = which.min(scores)

cat("\n--- Best Model Parameters ---\n")
cat("model: ", model, "\n")
cat("AIC score: ", min(scores), "\n")

# After printing, best model (as simple as possible) is AR(1) or MA(1)
print("AR(1) parameter: ")
print(model_ar1)
print("MA(1) parameter: ")
print(model_ma1)