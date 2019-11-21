library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(gridExtra)
library(xts)
library(feather)

# Set working dir
setwd("C:/Projects/tidal_flat_0d")

# Read data from csv
pressure = read.csv("data/interim/sutarkhali_pressure.csv")

# Filter NaNs and normalize by mean
pressure = pressure %>%
  filter(Pressure != "NaN") %>%
  filter(Datetime != "NA") %>%
  mutate(Pressure = Pressure - mean(Pressure)) %>%
  mutate(Datetime = mdy_hms(Datetime, tz = "Asia/Dhaka")) %>%
  as_tsibble(index = Datetime)


tides.sl = as.sealevel(elevation = pressure$Pressure, time = pressure$Datetime)
mod = tidem(t = tides.sl)

run_length = 50 #years
dt = "3 hours"
index = seq.POSIXt(pressure$Datetime[1], pressure$Datetime[1] + years(run_length), by = dt)

slr = .003
end_sl = run_length * slr
sl_vec = seq(0, end_sl, length.out = length(index))


tides = tibble(Datetime = index) %>%
  as_tsibble(index = Datetime) %>%
  arrange(Datetime)
  
tides$pressure = predict(mod, newdata=index)

tides$pressure = tides$pressure + sl_vec

write_feather(tides,'./data/interim/tides.feather')

pdata = read.csv("out.csv")

library(akima)
library(fields)
s = interp(pdata$ssc, pdata$slr, pdata$inundation_days)
image.plot(s)
