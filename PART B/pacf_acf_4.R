library(zoo)

# From STEP 1-3
df <- read.csv("PART B/Bitcoin.csv") 
df$observation_date <- as.Date(df$observation_date)
df <- df[order(df$observation_date), ] 
df$CBBTCUSD <- na.locf(df$CBBTCUSD, na.rm = FALSE)

log_returns <- diff(log(df$CBBTCUSD))

png("fig_03_acf_pacf.png", width = 800, height = 600, res = 100)


par(mfrow=c(2,1), mar=c(4,4,2,1))

acf(log_returns, 
    main = "Autocorrelation Function (ACF) of Bitcoin Returns", 
    lag.max = 40,  # Look at the last 40 days
    col = "blue",
    lwd = 2)

pacf(log_returns, 
     main = "Partial Autocorrelation Function (PACF) of Bitcoin Returns", 
     lag.max = 40,
     col = "red",
     lwd = 2)

par(mfrow=c(1,1))
dev.off()