library(zoo)

df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)

log_returns <- diff(log(df$CBBTCUSD))
dates <- df$observation_date[-1]

# Train/test split with the last 30 days for test
n <- length(log_returns)
test_size <- 30
train_size <- n - test_size

train_data <- log_returns[1:train_size]
test_data  <- log_returns[(train_size + 1):n]
test_dates <- dates[(train_size + 1):n]

# Train the model ONLY on the training set
model_ar1_train <- arima(train_data, order = c(0, 0, 1))

# Forecast the last unobserved 30 days 
forecast_result <- predict(model_ar1_train, n.ahead = test_size)
predicted_returns <- forecast_result$pred
se_returns <- forecast_result$se

# Confidence interval
lower_bound <- predicted_returns - 1.96 * se_returns
upper_bound <- predicted_returns + 1.96 * se_returns

png("fig_06_forecast.png", width = 800, height = 400, res = 100)

# Show the last 60 days of training + 30 days of test
plot_start <- train_size - 60
plot_dates <- dates[plot_start:n]
plot_actual <- log_returns[plot_start:n]

plot(plot_dates, plot_actual, type = "l", col = "darkgray", lwd = 1.5,
     main = "AR(1) Backtest: Predicted vs Actual (Last 30 Days)",
     xlab = "Date", ylab = "Log Return")

grid(col = "lightgray", lty = "dotted")
abline(h = 0, col = "black")


lines(test_dates, predicted_returns, col = "red", lwd = 2)


lines(test_dates, lower_bound, col = "blue", lty = 2)
lines(test_dates, upper_bound, col = "blue", lty = 2)

legend("topleft", legend = c("Actual Data", "Forecast", "95% Confidence"),
       col = c("darkgray", "red", "blue"), lty = c(1, 1, 2), lwd = 2)

dev.off()


rmse <- sqrt(mean((test_data - predicted_returns)^2))
cat("RMSE: ", rmse, "\n")