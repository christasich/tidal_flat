library(tsibble)
library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(arrow)

# Set working dir
args<-commandArgs(TRUE)

# Read data from csv
pressure = read_csv(args[1])


tides.sl = as.sealevel(elevation = pressure$Pressure, time = pressure$Datetime)
mod = tidem(t = tides.sl)

dt = args[2]
index = seq.POSIXt(as.POSIXct("2019-01-01 00:00:00", tz="Asia/Dhaka"), as.POSIXct("2019-12-31 23:59:59", tz="Asia/Dhaka"), by = dt)


tides = tibble(Datetime = index) %>%
  as_tsibble(index = Datetime) %>%
  arrange(Datetime)

tides$pressure = predict(mod, newdata=index)

write_feather(tides, args[3])

print('Tides made')

