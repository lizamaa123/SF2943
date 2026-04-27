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

df_clean <- data.frame(
  date = data_log$date,
  SE3 = as.numeric(ts_clean_val)
)

fig_orig <- plot_ly(data_orig, x = ~date, y = ~SE3, type = 'scatter', mode = 'lines', 
                    line = list(color = 'black', width = 1), name = 'Original (Dirty)')

fig_log <- plot_ly(data_log, x = ~date, y = ~SE3_log, type = 'scatter', mode = 'lines', 
                     line = list(color = 'steelblue', width = 1), name = 'Log Transformed')

interactive_plot <- subplot(fig_orig, fig_log, nrows = 2, shareX = TRUE, titleY = FALSE) %>%
  layout(
    title = 'SE3 Electricity Consumption: Original vs. Log-Transformed',
    height = 700,
    showlegend = TRUE,
    hovermode = "x unified"
  )

saveWidget(interactive_plot, "se3_data_comparison_log.html", selfcontained = TRUE)
browseURL("se3_data_comparison_log.html")