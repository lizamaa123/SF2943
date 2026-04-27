# help taken from: https://plotly.com/ggplot2/plot-data-from-csv/

#install.packages("plotly", repos = "https://cloud.r-project.org")
#install.packages("ggplot2", repos = "https://cloud.r-project.org")
library(htmlwidgets)
library(plotly)
library(ggplot2)

file_path = "dataset/se_d_data.csv"
data <- read.csv(file_path)
data$date <- as.Date(data$date, format = "%Y-%m-%d")

p <- ggplot(data = data, aes(x = date, y = SE3)) +
  geom_line(color = "steelblue", linewidth = 0.5) +
  labs(title = 'Daily Electricity Consumption in SE3', x = "Date", y = "Consumption") +
  theme_minimal()

interactive_plot <- ggplotly(p)
saveWidget(interactive_plot, "se3_data_init.html", selfcontained = TRUE)
browseURL("se3_data_init.html")