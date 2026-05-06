library(zoo)
library(tseries) 

# Fromm STEP 1 and 2
df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)

raw_prices <- df$CBBTCUSD

# Calculating log-returns
# ln(price_today) - ln(price_yesterday)
log_returns <- diff(log(raw_prices))

# Differencing reduces dataset by 1 day (1 day step)
dates_for_returns <- df$observation_date[-1]

# Check for unit roots (if p-value less than 0.05 then we can reject null hypothesis
# that data is not stationary)
adf_result <- adf.test(log_returns, alternative = "stationary")
print(adf_result)

if(adf_result$p.value < 0.05) {
  cat("\nResult: p-value < 0.05. The data is STATIONARY (Ready for ARMA!).\n")
} else {
  cat("\nResult: The data is NOT stationary.\n")
}

png("fig_02_stationary_log_returns.png", width = 800, height = 400, res = 100)

plot(dates_for_returns, log_returns, 
     type = "l", 
     col = "purple", 
     lwd = 1.0,
     main = "Step 2: Bitcoin Daily Log Returns",
     ylab = "Log Return", 
     xlab = "Date")

# Add a red dashed line exactly at 0 to show the constant mean
abline(h = 0, col = "red", lty = 2, lwd = 1.5)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")

dev.off()