library(htmlwidgets)
library(plotly)
library(forecast)

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

# Now we have THREE seasonal columns to grab
df_decomp <- data.frame(
  Date = data_log$date,
  Original = as.numeric(ts_clean_val),
  Trend = as.numeric(decomp$time.series[, "trend"]),
  Seasonal = as.numeric(decomp$time.series[, "seasonal"]),
  Residual = as.numeric(decomp$time.series[, "remainder"])
)

fig_data <- plot_ly(df_decomp, x = ~Date, y = ~Original, type = 'scatter', mode = 'lines', 
                    line = list(color = 'black', width = 1), name = 'Original')

fig_trend <- plot_ly(df_decomp, x = ~Date, y = ~Trend, type = 'scatter', mode = 'lines', 
                     line = list(color = 'red', width = 1), name = 'Trend')

fig_season <- plot_ly(df_decomp, x = ~Date, y = ~Seasonal, type = 'scatter', mode = 'lines', 
                      line = list(color = 'blue', width = 1), name = 'Weekly (7)')

fig_resid <- plot_ly(df_decomp, x = ~Date, y = ~Residual, type = 'scatter', mode = 'lines', 
                     line = list(color = 'steelblue', width = 1), name = 'Residual')

interactive_plot <- subplot(fig_data, fig_trend, fig_season, fig_resid, nrows = 4, shareX = TRUE, titleY = FALSE) %>%
  layout(
    title = 'STL Decomposition: 7-Day Cycles',
    height = 700,         
    showlegend = FALSE,     
    hovermode = "x unified"
  )

saveWidget(interactive_plot, "se3_stl_decomp_7_log.html", selfcontained = TRUE)
browseURL("se3_stl_decomp_7_log.html")

noise <- decomp$time.series[, "remainder"]
noise_mean <- mean(noise, na.rm = TRUE)
noise_variance <- var(noise, na.rm = TRUE)

cat("Noise Statistics \n")
cat("Mean:     ", round(noise_mean, 4), "\n")
cat("Variance: ", round(noise_variance, 4), "\n")