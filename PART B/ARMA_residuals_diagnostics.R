library(zoo)

# From STEP 1-2
df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)
log_returns <- diff(log(df$CBBTCUSD))

model_arma46 <- arima(log_returns, order = c(2, 0, 0))

# Extract residuals from model
model_residuals <- residuals(model_arma46)

cat("\n--- Ljung-Box Test on ARMA(1,1) Residuals ---\n")
lb_test <- Box.test(model_residuals, lag = 10, type = "Ljung-Box")
print(lb_test)

if(lb_test$p.value > 0.05) {
  cat("\nResult: p-value > 0.05. We fail to reject the null hypothesis.")
  cat("\nSUCCESS: The residuals are White Noise! The model is valid.\n")
} else {
  cat("\nResult: p-value < 0.05. The residuals still contain patterns.\n")
}

png("fig_04_ARMA_residual_diagnostics.png", width = 800, height = 600, res = 100)
par(mfrow=c(2,1), mar=c(4,4,2,1))

plot(model_residuals, type = "l", col = "darkgray", lwd = 1,
     main = "ARMA(1,1) Model Residuals", ylab = "Error", xlab = "Time")
abline(h = 0, col = "red", lty = 2)

acf(model_residuals, main = "ACF of ARMA(1,1) Residuals", lag.max = 40, col = "blue")

par(mfrow=c(1,1))
dev.off()