library(lubridate)
library(tidyverse)
library(tsibble)
library(oce)
library(gridExtra)
library(xts)

# Set working dir
setwd("C:/Projects/tidal_flat_0d")


# Read data from csv
pressure = read.csv("data/interim/sutarkhali_pressure.csv")
ssc = read.csv("data/interim/sutarkhali_ssc.csv")

# Filter NaNs and normalize by mean
pressure = pressure %>%
  filter(Pressure != "NaN") %>%
  filter(Datetime != "NA") %>%
  mutate(Pressure = Pressure - mean(Pressure)) %>%
  mutate(Datetime = mdy_hms(Datetime, tz = "Asia/Dhaka")) %>%
  as_tsibble(index = Datetime)

ssc = ssc %>%
  mutate(Datetime = mdy_hms(Datetime, tz = "Asia/Dhaka")) %>%
  mutate(Datetime = round_date(Datetime, unit = "1 minute")) %>%
  as_tsibble(index = Datetime)


tides.sl = as.sealevel(elevation = pressure$Pressure, time = pressure$Datetime)
mod = tidem(t = tides.sl)


index = seq.POSIXt(pressure$Datetime[1], tail(ssc$Datetime,1), by = "min")
tides = tibble(Datetime = index) %>%
  full_join(pressure, by = "Datetime") %>%
  as_tsibble(index = Datetime)

tides$Estimated.Pressure = predict(mod, newdata=tides$Datetime)

# Find high and low tides

window_size = 3 # in hours

high_tides = tides %>%
  filter(Estimated.Pressure > lead(Estimated.Pressure) & Estimated.Pressure > lag(Estimated.Pressure)) %>%
  mutate(high_tide = Estimated.Pressure) %>%
  mutate(high_low_cycle = "high")

low_tides = tides %>%
  filter(Estimated.Pressure < lead(Estimated.Pressure) & Estimated.Pressure < lag(Estimated.Pressure)) %>%
  mutate(low_tide = Estimated.Pressure) %>%
  mutate(high_low_cycle = "low")

combined_high_low_tides = as_tibble(bind_rows(high_tides, low_tides)) %>%
  arrange(Datetime) %>%
  mutate(amp = abs(Estimated.Pressure - lag(Estimated.Pressure, 1)))

# Find spring and neap tides

spring_tides = combined_high_low_tides %>%
  mutate(ma = slide_dbl(amp, ~ max(.,na.rm = TRUE), .size = 10, .align="center-right", .partial = TRUE)) %>%
  mutate(dt = lead(Datetime) - Datetime) %>%
  mutate(spring_neap_cycle = "spring") %>%
  mutate(spring_time = Datetime) %>%
  filter(ma == amp) %>%
  distinct()

neap_tides = combined_high_low_tides %>%
  mutate(ma = slide_dbl(amp, ~ min(.,na.rm = TRUE), .size = 10, .align="center-right", .partial = TRUE)) %>%
  mutate(dt = lead(Datetime) - Datetime) %>%
  mutate(spring_neap_cycle = "neap") %>%
  mutate(neap_time = Datetime) %>%
  filter(ma == amp) %>%
  distinct()

combined_spring_neap_tides = as_tibble(bind_rows(spring_tides, neap_tides)) %>%
  arrange(Datetime) %>%
  subset(select = c(Datetime, Pressure, Estimated.Pressure, amp, spring_neap_cycle, spring_time, neap_time))

data = full_join(tides, ssc, by = "Datetime") %>%
  full_join(subset(combined_high_low_tides, select = c(Datetime, high_tide, low_tide, amp, high_low_cycle)), by = "Datetime") %>%
  full_join(subset(combined_spring_neap_tides, select = c(Datetime, spring_neap_cycle, spring_time, neap_time)), by = "Datetime") %>%
  arrange(Datetime) %>%
  as_tsibble(index = Datetime) %>%
  mutate(high_tide = na.locf(high_tide, na.rm = FALSE)) %>%
  mutate(low_tide = na.locf(low_tide, na.rm = FALSE)) %>%
  mutate(amp = na.locf(amp, na.rm = FALSE)) %>%
  mutate(month = month(Datetime, label = TRUE)) %>%
  mutate(week = week(Datetime)) %>%
  mutate(spring_time = na.locf(spring_time, na.rm = FALSE)) %>%
  mutate(neap_time = na.locf(neap_time, na.rm = FALSE)) %>%
  mutate(dt_spring = as.numeric(Datetime - spring_time, units = "hours")) %>%
  mutate(dt_neap = as.numeric(Datetime - neap_time, units = "hours")) %>%
  filter(SSC != "NA")

# boxes for plotting spring neap

# spring_box = spring_tides %>%
#   mutate(xmin = (Datetime + dt) - days(2)) %>%
#   mutate(xmax = (Datetime + dt) + days(2)) %>%
#   subset(select = c(xmin, xmax)) %>%
#   mutate(ymin = -Inf) %>%
#   mutate(ymax = Inf)
# 
# neap_box = neap_tides %>%
#   mutate(xmin = (Datetime + dt) - days(2)) %>%
#   mutate(xmax = (Datetime + dt) + days(2)) %>%
#   subset(select = c(xmin, xmax)) %>%
#   mutate(ymin = -Inf) %>%
#   mutate(ymax = Inf)
# 
# 
# start = as.POSIXct('2015-05-1')
# stop = as.POSIXct('2015-5-30')
# 
# plot_data = tides %>%
#   filter(Datetime >= start, Datetime <= stop) %>%
#   sample_frac(0.2)
# 
# ggplot() +
#   geom_point(data = plot_data, aes(x = Datetime, y = Pressure), size = 1, alpha = 0.1) +
#   geom_rect(data=spring_box, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
#             color="red",
#             fill="red",
#             alpha=0.3) +
#   geom_rect(data=neap_box, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
#             color="blue",
#             fill="blue",
#             alpha=0.3) +
#   scale_x_datetime(expand = c(0, 0), limits = c(start, stop)) + 
#   scale_y_continuous(limits = c(-3, 3)) +
#   theme_bw()
# 
# data = data %>%
#   filter(SSC != "NA")
# 
# ggplot(data = data) +
#   geom_boxplot(aes(y = SSC, x = week, group = week))
# 
# ggplot(data = data) +
#   geom_boxplot(aes(y = SSC, x = round(dt_spring,0), group = round(dt_spring,0))) +
#   scale_y_continuous(limits = c(0, 0.5))

# Plot SSC vs

plot_data = data %>%
  sample_frac(0.05, replace = FALSE)

p1 = ggplot(plot_data, aes(y = SSC, x = dt_spring)) +
  geom_point(alpha = 0.1) +
  geom_smooth() +
#  geom_line(aes(y = slide_dbl(SSC, ~ mean(.,na.rm = TRUE), .size = 1440, .align="center-right", .partial = TRUE)), color = "red") +
  scale_y_continuous(limits = c(0,1))


ggplotly(p1)


p2 = ggplot(plot_data) +
  geom_boxplot(aes(y = SSC, x = round(dt_spring/24,0), group = round(dt_spring/24,0))) +
  scale_y_continuous(limits = c(0, 0.5))

ggplotly(p2)

p3 = ggplot(plot_data) +
  geom_boxplot(aes(y = SSC, x = week, group = week)) +
  geom_smooth(aes(x = week, y = SSC)) +
  scale_y_continuous(limits = c(0, 0.5)) +
  scale_x_continuous(breaks = seq(0, 52, 8))

ggplotly(p3)




data %>%
  as_tibble() %>% 
  mutate(year = year(Datetime), month = as.integer(month)) %>%
  group_by(year, week) %>%
  summarize(SSC = mean(SSC, na.rm = TRUE)) %>%
  ungroup()  %>%
  ggplot(aes(x = week, y = SSC)) + 
  geom_smooth() +
  geom_point(alpha = 0.3) + 
  scale_x_continuous(breaks = seq(0, 52, 8))



data_by_week = data %>%
  as_tibble() %>% 
  mutate(year = year(Datetime), month = as.integer(month)) %>%
  group_by(year, week) %>%
  summarize(SSC = mean(SSC, na.rm = TRUE)) %>%
  ungroup()

loess_mod = loess(SSC ~ week, data_by_week)
out_data = tibble("week" = as.numeric(seq(1,53)))
out_data$ssc = predict(loess_mod, newdata = out_data$week)
write.csv(out_data, "./data/processed/ssc_by_week.csv", row.names = FALSE, quote = FALSE)
