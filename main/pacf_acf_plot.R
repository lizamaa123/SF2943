library(forecast)
library(ggplot2)
library(plotly)
library(htmlwidgets)

file_path = "dataset/se_d_data.csv"
data <- read.csv(file_path)
data$date <- as.Date(data$date, format = "%Y-%m-%d")

# Log-transformed data
# copy data and remove extreme outlier and aply log-transform
data_log <- subset(data, date < as.Date("2026-04-12"))
data_log$SE3_log <- log(data_log$SE3)
# create ts object
ts_log <- ts(data_log$SE3_log, frequency = 7)
ts_clean_val <- tsclean(ts_log)

# STL Decomposition (X = trend + seasonality + noise)
decomp <- stl(ts_clean_val, s.window = "periodic")

# Extract just the stationary noise = remainder
noise <- decomp$time.series[, "remainder"]

# Create ACF and PACF plots with 95% conf. int.
p_acf <- ggAcf(noise) + 
  theme_minimal() + 
  labs(title = "ACF of noise")

p_pacf <- ggPacf(noise) + 
  theme_minimal() + 
  labs(title = "PACF of noise")

interactive_plot <- subplot(ggplotly(p_acf), ggplotly(p_pacf), nrows = 2, titleY = TRUE) %>%
  layout(
    height = 800, 
    showlegend = FALSE,
    title = "Investigating Stationarity (Log-transformed): ACF and PACF"
  )

saveWidget(interactive_plot, "se3_acf_pacf_log.html", selfcontained = TRUE)
browseURL("se3_acf_pacf_log.html")

cat("Auto.ARIMA\n")
fit_auto <- auto.arima(noise, seasonal = FALSE, stepwise = FALSE, approximation = FALSE, trace = TRUE)
summary(fit_auto)

cat("\nTesting Manual Hypotheses \n")
fit_54 <- Arima(noise, order = c(5, 0, 4))
fit_53 <- Arima(noise, order = c(5, 0, 3))
fit_33 <- Arima(noise, order = c(3, 0, 3))
fit_32 <- Arima(noise, order = c(3, 0, 2))
fit_43 <- Arima(noise, order = c(4, 0, 3))

cat("\nFINAL RESULTS\n")
print(paste("Auto Model AICc:", fit_auto$aicc))
print(paste("ARMA(5,4) AICc: ", fit_54$aicc))
print(paste("ARMA(5,3) AICc: ", fit_53$aicc))
print(paste("ARMA(3,3) AICc: ", fit_33$aicc))
print(paste("ARMA(3,2) AICc: ", fit_32$aicc))
print(paste("ARMA(4,3) AICc: ", fit_43$aicc))
