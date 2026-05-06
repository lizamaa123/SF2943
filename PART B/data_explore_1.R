# Loading dataset (obs. date / CBBTCUSD = coinbase bitcoin)
df <- read.csv("PART B/Bitcoin.csv")

# Sorting chronologically dep. of date
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ]

# Extracting target variable
raw_prices <- df$CBBTCUSD

print(summary(df))

plot(df$observation_date, raw_prices, 
     type = "l", 
     col = "blue", 
     lwd = 1.5,
     main = "Raw Bitcoin Prices (CBBTCUSD)",
     ylab = "Price (USD)", 
     xlab = "Date")
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
dev.off()