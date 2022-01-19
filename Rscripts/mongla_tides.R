library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(arrow)
library(feather)
library(imputeTS)
library(dygraphs)
library(xts)
library(forecast)
library(feasts)
library(purrr)

setwd("~/projects/tidal_flat_0d")

data <- read.csv("data/raw/mongla_tides.csv") |>
  mutate(datetime = dmy_hms(datetime, tz = "Asia/Dhaka")) |>
  distinct(datetime, .keep_all = TRUE) |>
  rename(elevation = height) |>
  mutate(elevation = elevation / 100) |>
  as_tsibble(index = datetime) |>
  fill_gaps()

longitude <- 89.6
latitude <- 22.4833

start <- as.POSIXct("1977-01-01", tz = "Asia/Dhaka")
end <- start + days(365+120)
subset <- data |>
  filter_index(toString(start)~toString(end))

tides.sl <- as.sealevel(
  elevation = subset$elevation,
  time = subset$datetime,
  stationName = "Mongla",
  region = "Bangladesh",
  longitude = longitude,
  latitude = latitude,
  GMTOffset = 6
)
base_mod <- tidem(t = tides.sl)

const <- data.frame(
  year = year(start),
  name = base_mod@data$name, 
  freq = base_mod@data$freq,
  amp = base_mod@data$amplitude, 
  phase = base_mod@data$phase,
  p_value = base_mod@data$p)|>
  as_tsibble(index = year, key = name)

for (i in 1978:2010){
  start <- as.POSIXct(paste(i, "-01-01", sep=""), tz = "Asia/Dhaka")
  end <- start + days(365+120)
  subset <- data |>
    filter_index(toString(start)~toString(end))
  
  tides.sl <- as.sealevel(
    elevation = subset$elevation,
    time = subset$datetime,
    stationName = "Mongla",
    region = "Bangladesh",
    longitude = longitude,
    latitude = latitude,
    GMTOffset = 6
  )
  mod <- tidem(t = tides.sl)
  
  df <- data.frame(
      year = year(start),
      name = mod@data$name, 
      freq = mod@data$freq,
      amp = mod@data$amplitude, 
      phase = mod@data$phase,
      p_value = mod@data$p)|>
      as_tsibble(index = year, key = name)
  const <- bind_rows(const, df)
}

models <- const |>
  as_tibble() |>
  nest(-name) |>
  mutate(
    m = map(data, loess, formula = amp ~ year, control=loess.control(surface="direct"))
  ) |>
  select(name, m)

multi <- function(df, n) {
  varname <- toString(n)
  mutate(df, !!varname := predict(m[[1]], n))
}

for (i in 1977:2020){
  amps <- multi(const_models, i)
}

const |>
  filter(amp > 0.2) |>
  autoplot(.vars = amp)

amps <- amps  |>
  select(-m) |>
  pivot_longer(!name, names_to = "year", values_to = "amp") |>
  mutate(year = as.numeric(year)) |>
  as_tsibble(index = year, key = name)

remove(preds)
for (i in 1977:2011){
  vals <- amps |>
    filter_index(i)
  
  mod <- as.tidem(
    tRef = base_mod@data$tRef,
    latitude = latitude,
    name = vals$name,
    amplitude = vals$amp,
    phase = base_mod@data$phase
  )
  
  start <- as.POSIXct(paste(i, "-01-01", sep = ""), tz = "Asia/Dhaka")
  end <- as.POSIXct(paste(i, "-12-31", sep = ""), tz = "Asia/Dhaka")
  new <- tsibble(datetime = seq.POSIXt(start, end, by = dt), index = datetime) %>%
    mutate(elevation = predict(mod, newdata = .$datetime))
  preds <- bind_rows(if(exists("preds")) preds, new)
}

combined <- data |>
  left_join(preds, by = "datetime") |>
  rename(observed = elevation.x, predicted = elevation.y) |>
  mutate(predicted = predicted + (observed[1] - predicted[1]))

plt_data <- combined |>
  as_tibble() %>%
  {. ->> tmp} |>
  select(observed, predicted) %>%
  as.xts(order.by=tmp$datetime)

# Finally the plot
dygraph(plt_data, xlab = "Year", ylab = "Elevation") |>
  # dyAxis("y", label = "Elevation (m)") |>
  # dyAxis("y2", label = "Error (m)", independentTicks = TRUE) |>
  dySeries("observed", label = "Observed", color = "royalblue") |>
  dySeries("predicted", label = "Predicted", color = "indianred") |>
  dyOptions(labelsUTC = TRUE, fillGraph = TRUE, drawGrid = FALSE) |>
  dyRangeSelector() |>
  dyCrosshair(direction = "both") |>
  dyHighlight(highlightCircleSize = 5, highlightSeriesBackgroundAlpha = 0.5, hideOnMouseOut = FALSE)

# data |>
#   filter_index("2011-01-01"~"2011-01-30") |>
#   # filter_index("2011-07-01"~"2011-07-31") |>
#   ggplot(aes(x=datetime)) +
#   geom_line(aes(y=elevation, color="Observed"), alpha=0.5) +
#   geom_line(aes(y=predicted, color="Predicted"), alpha=0.5) +
#   scale_colour_manual(values = c("Observed" = "royalblue", "Predicted" = "indianred"))




# write_feather(tides, 'data/interim/mongla_tides.feather')
# 
# combined |> 
#   # filter(elevation > 0) |>
#   mutate(diff = predicted-elevation) |>
#   pull(diff) |>
#   sum
# 
# 
# year <- 2020
# years <- 1
# dt <- '5 min'
# 
# index <- seq.POSIXt(as.POSIXct(paste(year, "-01-01 00:00:00", sep = ""), tz = "Asia/Dhaka"), as.POSIXct(paste(year, "-12-31 23:59:59", sep = ""), tz = "Asia/Dhaka"), by = dt)
# tides <- tibble(datetime = index) |>
#   as_tsibble(index = datetime) |>
#   arrange(datetime)
# tides$elevation <- predict(mod, newdata = index)


# for (i in 1:(years-2)) {
#   year <- year + 1
#   index <- seq.POSIXt(as.POSIXct(paste(year, "-01-01 00:00:00", sep = ""), tz = "Asia/Dhaka"), as.POSIXct(paste(year, "-12-31 23:59:59", sep = ""), tz = "Asia/Dhaka"), by = dt)
#   new <- tibble(datetime = index) |>
#     as_tsibble(index = datetime) |>
#     arrange(datetime)
#   new$elevation <- predict(mod, newdata = index)
#   tides <- bind_rows(tides, new)
# }
# tides <- as_tsibble(tides)
# 
# write_feather(tides, 'data/interim/tides-50yr-10s.feather')