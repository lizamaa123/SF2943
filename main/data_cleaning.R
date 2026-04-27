library(htmlwidgets)
library(plotly)
library(forecast)

file_path = "dataset/se_d_data.csv"
# Original (uncleaned) data
data_orig <- read.csv(file_path)
data_orig$date <- as.Date(data_orig$date, format = "%Y-%m-%d")

# Edited (cleaned) data
# copy data
data_clean <- data_orig 
# remove last data point since outlier and drops to zero (incomplete data point)
data_clean <- subset(data_clean, date < as.Date("2026-04-12"))
# convert to time series object
ts_clean <- ts(data_clean$SE3)
# tsclean() removes any other hidden outlier
ts_clean_val <- tsclean(ts_clean)

df_clean <- data.frame(
  date = data_clean$date,
  SE3 = as.numeric(ts_clean_val)
)

fig_orig <- plot_ly(data_orig, x = ~date, y = ~SE3, type = 'scatter', mode = 'lines', 
                    line = list(color = 'black', width = 1), name = 'Original (Dirty)')

fig_clean <- plot_ly(df_clean, x = ~date, y = ~SE3, type = 'scatter', mode = 'lines', 
                     line = list(color = 'steelblue', width = 1), name = 'Cleaned')

interactive_plot <- subplot(fig_orig, fig_clean, nrows = 2, shareX = TRUE, titleY = FALSE) %>%
  layout(
    title = 'SE3 Electricity Consumption: Original vs. Cleaned Data',
    height = 700,
    showlegend = TRUE,
    hovermode = "x unified"
  )

saveWidget(interactive_plot, "se3_data_comparison.html", selfcontained = TRUE)
browseURL("se3_data_comparison.html")