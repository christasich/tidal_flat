library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(arrow)
library(feather)
library(imputeTS)
library(forecast)

setwd("~/projects/tidal_flat_0d")

data <- read.csv("data/interim/sutarkhali_ext.csv") |>
  drop_na() |>
  mutate(datetime = dmy_hms(datetime, tz = "Asia/Dhaka")) |>
  as_tsibble(index = datetime) |>
  filter_index("2014-11-03 17:10:00"~"2016-10-11 21:20:00")

count_gaps(data, .full = TRUE) |>
  mutate(missing_days = difftime(.to, .from, units="days"))

data <- data |>
  fill_gaps()

data |>
  ggplot(aes(x=datetime, y=elevation)) +
  geom_line()

daily <- data |>
  index_by(date = ~ as_date(.)) |>
  summarize(mean = mean(elevation, na.rm = TRUE))

daily |>
  ggplot(aes(x=date, y=mean)) +
  geom_line()

df <- daily$mean %>%
  ts(start = daily$date[1], frequency = 365.25)
decompose_df <- tslm(df ~ trend + fourier(df, 2))
trend <- coef(decompose_df)[1] + coef(decompose_df)["trend"]*seq_along(df)
components <- cbind(
  data = df,
  trend = trend,
  season = df - trend - residuals(decompose_df),
  remainder = residuals(decompose_df)
)
autoplot(components, facet=TRUE)

df <- data %>%
  .$elevation %>%
  ts(start = decimal_date(data$datetime[1]), frequency = 365.25 * 24 * 60 / 10)
decompose_df <- tslm(df ~ trend + fourier(df, 2))
trend <- coef(decompose_df)[1] + coef(decompose_df)["trend"]*seq_along(df)
components <- cbind(
  data = df,
  trend = trend,
  season = df - trend - residuals(decompose_df),
  remainder = residuals(decompose_df)
)
autoplot(components, facet=TRUE)


weekly <- data |>
  index_by(week = ~ yearweek(.)) |>
  summarize(mean = mean(elevation, na.rm = TRUE)) |>
  mutate(date = as.Date(week))

weekly |>
  ggplot(aes(x=week, y=mean)) +
  geom_line()

monthly <- data |>
  index_by(month = ~ yearmonth(.)) |>
  summarize(mean = mean(elevation, na.rm = TRUE)) |>
  mutate(date = as.Date(month))

monthly |>
  ggplot(aes(x=month, y=mean)) +
  geom_line()


periods <- c(365.25 * 24 * 60 / 10)
ser <- msts(data = data$elevation, seasonal.periods = periods, start=decimal_date(data$datetime[1]))
stl_data <- mstl(ser, robust = TRUE)
autoplot(stl_data)

seasonal <- seasonal(stl_data) |>
  as_tsibble(index = daily$date) |>
  index_by(year_day = ~ yday(.)) |>
  summarize(adjustment = mean(value))

seasonal <- seasonal(stl_data) |>
  as_tsibble(index = weekly$week) |>
  index_by(week = ~ week(.)) |>
  summarize(adjustment = mean(value))

seasonal(stl_data) |>
  as_tsibble(index = weekly$week) |>
  ggplot(aes(x=index, y=value)) +
  geom_line()

start <- as.POSIXct("2015-01-01", tz = "Asia/Dhaka")
end <- start + days(365+120)

data <- data |>
  mutate(adjusted = elevation - seasonal)

longitude <- 89.4892
latitude <- 22.4992

sub_start <- as.POSIXct("2015-01-01", tz = "Asia/Dhaka")
sub_end <- sub_start + days(365+120)
subset <- data |>
  filter_index(toString(sub_start)~toString(sub_end))

tides.sl <- as.sealevel(
  elevation = subset$elevation,
  time = subset$datetime,
  stationName = "Sutarkhali",
  region = "Bangladesh",
  longitude = longitude,
  latitude = latitude,
  GMTOffset = 6
)

mod <- tidem(t = tides.sl)

const <- data.frame("name" = mod@data$name, 
                    "freq" = mod@data$freq,
                    "amp" = mod@data$amplitude, 
                    "phase" = mod@data$phase,
                    "p_value" = mod@data$p) |>
  as_tibble() |>
  arrange(desc(amp))

year <- 2020
years <- 50
dt <- '1 sec'

index <- seq.POSIXt(as.POSIXct(paste(year, "-01-01 00:00:00", sep = ""), tz = "Asia/Dhaka"), as.POSIXct(paste(year, "-12-31 23:59:59", sep = ""), tz = "Asia/Dhaka"), by = dt)
tides <- tsibble(datetime=index, index=datetime) |>
  mutate(elevation = predict(mod, newdata=datetime))

for (i in 1:(years-1)) {
  year <- year + 1
  index <- seq.POSIXt(as.POSIXct(paste(year, "-01-01 00:00:00", sep = ""), tz = "Asia/Dhaka"), as.POSIXct(paste(year, "-12-31 23:59:59", sep = ""), tz = "Asia/Dhaka"), by = dt)
  new <- tsibble(datetime = index, index=datetime) |>
    mutate(elevation = predict(mod, newdata=datetime))

  tides <- bind_rows(tides, new)
}

write_feather(tides, 'data/interim/tides-50yr-1s.feather')

multiconst <- function(df, n) {
  varname <- const$name[n]
  mutate(df, !!varname := const$amp[n] * cos(const$freq[n] * df$time + const$phase[n]))
}

tides <- subset |>
  mutate(model = predict(mod)) |>
  mutate(time = as.numeric(difftime(datetime, mod@data$tRef, units="mins")) / 10)

const <- data.frame("name" = mod@data$name, 
                    "freq" = mod@data$freq,
                    "amp" = mod@data$amplitude, 
                    "phase" = mod@data$phase,
                    "p_value" = mod@data$p) |>
  as_tibble() |>
  arrange(desc(amp))

for (i in 1:length(const$name)) {
  tides <- multiconst(tides, i)
}

tides <- tides |>
  as_tibble() %>%
  mutate(cos = rowSums(select(., -datetime, -elevation, -model, -time))) |>
  as_tsibble(index = datetime) |>
  select(datetime, elevation, model, cos, SA)

library(scales)

tides |>
  sample_frac(0.001) |>
  ggplot(aes(x=datetime)) +
  geom_line(aes(y=elevation), color="blue")

year <- 2020
for (i in 1:(years-1)) {
  print(year+i)
}
