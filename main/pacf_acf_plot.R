library(forecast)
library(ggplot2)
library(plotly)
library(htmlwidgets)

file_path = "dataset/se_d_data.csv"
data <- read.csv(file_path)
data$date <- as.Date(data$date, format = "%Y-%m-%d")

# Edited (cleaned) data
# copy data 
# remove last data point since outlier and drops to zero (incomplete data point)
data_clean <- subset(data, date < as.Date("2026-04-12"))
# convert to time series object
ts_clean <- ts(data_clean$SE3, frequency = 7)
# tsclean() removes any other hidden outlier
ts_clean_val <- tsclean(ts_clean)

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
    title = "Investigating Stationarity: ACF and PACF"
  )

saveWidget(interactive_plot, "se3_acf_pacf.html", selfcontained = TRUE)
browseURL("se3_acf_pacf.html")

cat("Auto.ARIMA\n")
fit_auto <- auto.arima(noise, seasonal = FALSE, stepwise = FALSE, approximation = FALSE, trace = TRUE)

cat("\nTesting Manual Hypotheses \n")
fit_54 <- Arima(noise, order = c(5, 0, 4))
fit_53 <- Arima(noise, order = c(5, 0, 3))
fit_33 <- Arima(noise, order = c(3, 0, 3))

cat("\nFINAL RESULTS\n")
print(paste("Auto Model AICc:", fit_auto$aicc))
print(paste("ARMA(5,4) AICc: ", fit_54$aicc))
print(paste("ARMA(5,3) AICc: ", fit_53$aicc))
print(paste("ARMA(3,3) AICc: ", fit_33$aicc))