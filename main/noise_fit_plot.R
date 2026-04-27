library(forecast)
library(plotly)
library(htmlwidgets)

file_path = "dataset/se_d_data.csv"
# Original (uncleaned) data
data_orig <- read.csv(file_path)
data_orig$date <- as.Date(data_orig$date, format = "%Y-%m-%d")

# Log-transformed data
# copy data and remove extreme outlier and aply log-transform
data_log <- subset(data_orig, date < as.Date("2026-04-12"))
data_log$SE3_log <- log(data_log$SE3)
# create ts object
ts_log <- ts(data_log$SE3_log, frequency = 7)
ts_clean_val <- tsclean(ts_log)

# STL Decomposition (X = trend + seasonality + noise)
decomp <- stl(ts_clean_val, s.window = "periodic")

# Extract just the stationary noise = remainder
noise <- decomp$time.series[, "remainder"]

# Fit the explicit ARMA(5,4) or ARIMA(5,0,4) model, since least AICc value
noise_model <- Arima(noise, order = c(5, 0, 4))

# extracting the models "fitted value" = i.e what it has learned
fitted_noise <- fitted(noise_model)

df_plot <- data.frame(
  Date = data_log$date,
  Actual_Noise = as.numeric(noise),
  Fitted_Model = as.numeric(fitted_noise)
)

fig <- plot_ly(df_plot, x = ~Date) %>%
  add_lines(y = ~Actual_Noise, name = "Actual Noise", line = list(color = "black", width = 1, opacity = 0.5)) %>%
  add_lines(y = ~Fitted_Model, name = "ARMA(5,4) Fitted", line = list(color = "red", width = 1.5)) %>%
  layout(title = "Actual Noise vs. ARMA(5,4) Fitted Values (Log-Transformed)", 
         hovermode = "x unified",
         yaxis = list(title = "Noise Level"))

saveWidget(fig, "fitted_noise_overlay_log.html", selfcontained = TRUE)
browseURL("fitted_noise_overlay_log.html")

# Diagnostics
summary(noise_model)
checkresiduals(noise_model)