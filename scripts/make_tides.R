library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(feather)

# Set working dir
setwd("C:/Projects/tidal_flat_0d")
args<-commandArgs(TRUE)

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

run_length = as.numeric(args[1]) #years
dt = args[2]
slr = as.numeric(args[3])

index = seq.POSIXt(pressure$Datetime[1], pressure$Datetime[1] + years(run_length), by = dt)
end_sl = run_length * slr
sl_vec = seq(0, end_sl, length.out = length(index))


tides = tibble(Datetime = index) %>%
  as_tsibble(index = Datetime) %>%
  arrange(Datetime)
  
tides$pressure = predict(mod, newdata=index)

tides$pressure = tides$pressure + sl_vec

write_feather(tides,sprintf('./data/interim/feather/tides/tides_%s_slr', slr))
