library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(arrow)
library(feather)
library(imputeTS)
library(forecast)

setwd("~/projects/tidal_flat_0d")

data <- read.csv("data/interim/sutarkhali_pressure.csv") |>
  drop_na() |>
  rename(datetime = Datetime, elevation = Pressure) |>
  mutate(datetime = dmy_hms(datetime, tz = "Asia/Dhaka")) |>
  as_tsibble(index = datetime) |>
  fill_gaps() |>
  mutate(date = as.Date(datetime))

ggplot(data, aes(x=elevation)) +
  geom_density()

periods <- c(365.25 * 24 * 60 / 10)
ser <- msts(data = data$elevation, seasonal.periods = periods, ts.frequency = periods, start = decimal_date(data$datetime[1]))

stl_data <- mstl(ser)
autoplot(stl_data)

seasonal <- seasonal(stl_data)
trend <- trendcycle(stl_data)

data <- data |>
  mutate(adjusted = elevation - seasonal)

longitude <- 89.4892
latitude <- 22.4992

sub_start <- as.POSIXct("2015-01-01", tz = "Asia/Dhaka")
sub_end <- sub_start + days(365+120)
subset <- data |>
  filter_index(toString(sub_start)~toString(sub_end))

tides.sl <- as.sealevel(
  elevation = data$adjusted,
  time = data$datetime,
  stationName = "Sutarkhali",
  region = "Bangladesh",
  longitude = longitude,
  latitude = latitude,
  GMTOffset = 6
)
mod <- tidem(t = tides.sl)
summary(mod)

dt <- "10 min"

start <- as.POSIXct(head(data$datetime, 1), tz = "Asia/Dhaka")
end <- as.POSIXct(tail(data$datetime, 1))
index <- seq.POSIXt(start, end, by = dt)
vals <- predict(mod, newdata = index)

data <- data |>
  mutate(predicted = vals + seasonal + trend) |>
  mutate(error = elevation - predicted)

plt_data <- data |>
  filter_index("2011-07-01"~"2011-07-31") |>
  as_tibble() %>%
  {. ->> tmp} |>
  select(elevation, predicted, error) %>%
  as.xts(order.by=tmp$datetime)

# Finally the plot
dygraph(plt_data[,1:2], xlab = "Year", ylab = "Elevation") |>
  dyAxis("y", label = "Elevation (m)") |>
  dyAxis("y2", label = "Error (m)", independentTicks = TRUE) |>
  dySeries("elevation", label = "Observed", color = "royalblue") |>
  dySeries("predicted", label = "Predicted", color = "indianred") |>
  dyOptions(labelsUTC = TRUE, fillGraph = TRUE, drawGrid = FALSE) |>
  dyRangeSelector() |>
  dyCrosshair(direction = "both") |>
  dyHighlight(highlightCircleSize = 5, highlightSeriesBackgroundAlpha = 0.5, hideOnMouseOut = FALSE)
