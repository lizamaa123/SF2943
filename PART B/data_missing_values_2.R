library(zoo)

# From STEP 1
df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 

# LOCF
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)

raw_prices <- df$CBBTCUSD

png("fig_01_raw_bitcoin_prices.png")

plot(df$observation_date, raw_prices, 
     type = "l", 
     col = "blue", 
     lwd = 1.5,
     main = "Step 1: Cleaned Raw Bitcoin Prices",
     ylab = "Price (USD)", 
     xlab = "Date")

grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")

dev.off()