library(tsibble)
library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(arrow)
library(feather)

setwd('projects/tidal_flat_0d')


# Read data from csv
pressure = read.csv("data/interim/sutarkhali_pressure.csv")

pressure = pressure %>%
  filter(Pressure != "NaN") %>%
  filter(Datetime != "NaT") %>%
  mutate(Pressure = Pressure - mean(Pressure) - 0.29) %>%
  mutate(Datetime = dmy_hms(Datetime, tz = "Asia/Dhaka")) %>%
  as_tsibble(index = Datetime)


tides.sl = as.sealevel(elevation = pressure$Pressure, time = pressure$Datetime)
mod = tidem(t = tides.sl)

dt = '1 sec'
index = seq.POSIXt(as.POSIXct("2019-01-01 00:00:00", tz="Asia/Dhaka"), as.POSIXct("2019-12-31 23:59:59", tz="Asia/Dhaka"), by = dt)

tides = tibble(Datetime = index) %>%
  as_tsibble(index = Datetime) %>%
  arrange(Datetime)

tides$pressure = predict(mod, newdata=index)

# pos_tides = tides %>%
#   filter(Datetime > '2020-03-09 03:00:00' & Datetime < '2020-03-09 12:00:00') %>%
#   filter(pressure > 0)
# 
# ggplot(data=pos_tides, aes(Datetime, pressure)) +
#   geom_point()

write_feather(tides, 'data/interim/tides/tides-1yr-1s.feather')

print('Tides made')
