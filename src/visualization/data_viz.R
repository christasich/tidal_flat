library(feather)
library(tsibble)
library(ggplot2)
library(tidyverse)
library(xts)
library(lubridate)

model = 'base'
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df = read_feather(sprintf('../../data/interim/%s.feather', model)) %>%
  as_tsibble(index = Datetime)

dt = as.double((df$Datetime[2] - df$Datetime[1]), units = 'secs')

high_tides = df %>%
  filter(h > lead(h) & h > lag(h)) %>%
  mutate(high_tide = h) %>%
  mutate(high_low_cycle = "high") %>%
  mutate(ma_high_tide = slide_dbl(high_tide, ~ mean(.,na.rm = TRUE), .size = 1400, .align="right", .partial = TRUE))

low_tides = df %>%
  filter(h < lead(h) & h < lag(h)) %>%
  mutate(low_tide = h) %>%
  mutate(high_low_cycle = "low")

combined_high_low_tides = as_tibble(bind_rows(high_tides, low_tides)) %>%
  arrange(Datetime) %>%
  mutate(amp = abs(h - lag(h)))

data = df %>%
  full_join(subset(combined_high_low_tides, select = c(Datetime, high_tide, low_tide, amp, high_low_cycle)), by = "Datetime")

down = na.locf(data$high_tide, na.rm = FALSE)
up = na.locf(data$high_tide, na.rm = FALSE, fromLast=TRUE)
max = pmax(down, up, na.rm = TRUE)

data$high_tide = max

window = 3*24*60*60 / dt

spring_tides = combined_high_low_tides %>%
  mutate(ma = slide_dbl(amp, ~ max(.,na.rm = TRUE), .size = window, .align="center-right", .partial = TRUE)) %>%
  filter(ma == amp) %>%
  distinct() %>%
  mutate(spring_neap_cycle = "spring")

neap_tides = combined_high_low_tides %>%
  mutate(ma = slide_dbl(amp, ~ min(.,na.rm = TRUE), .size = window, .align="center-right", .partial = TRUE)) %>%
  filter(ma == amp) %>%
  distinct() %>%
  mutate(spring_neap_cycle = "neap")

combined_spring_neap_tides = as_tibble(bind_rows(spring_tides, neap_tides)) %>%
  arrange(Datetime) %>%
  subset(select = c(Datetime, h, amp, spring_neap_cycle))

data = data %>%
  full_join(subset(combined_spring_neap_tides, select = c(Datetime, spring_neap_cycle)), by = "Datetime")

spring_highs = data %>%
  filter(spring_neap_cycle == 'spring') %>%
  mutate(ma_spring = slide_dbl(high_tide, ~ mean(.,na.rm = TRUE), .size = 100, .align="right", .partial = TRUE))

monsoon_spring_highs = spring_highs %>%
  mutate(week = week(Datetime)) %>%
  filter(week > 32 & week < 45) %>%
  mutate(ma_monsoon_spring_highs = slide_dbl(high_tide, ~ mean(.,na.rm = TRUE), .size = 30, .align="right", .partial = TRUE))

# p = ggplot(data = data) +
#   geom_line(aes(x = Datetime, y = h), color = 'blue', alpha = 0.05) +
#   geom_line(aes(x = Datetime, y = z), color = 'black', size = 1) +
#   #geom_line(aes(x = Datetime, y = MHW), color = 'red') +
#   geom_smooth(aes(x = Datetime, y = high_tide), color = 'red') +
#   scale_x_datetime(expand = c(0, 0), limits = c(data$Datetime[1], tail(data$Datetime, n=1))) +
#   coord_cartesian(ylim=c(0,4.5)) +
#   labs(x = 'Year', y = 'Elevation (m)', title = model)
# 
# ggsave(sprintf('%s.png', model), plot=p)

ggplot(data = data) +
  geom_line(aes(x = Datetime, y = h), color = 'blue', alpha = 0.2) +
  geom_line(aes(x = Datetime, y = z), color = 'black', size = 1) +
  #geom_smooth(data = spring_highs, aes(x = Datetime, y = high_tide), color = 'red') +
  geom_line(data = spring_highs, aes(x = Datetime, y = ma_spring), color = 'red', size = 1) +
  #geom_smooth(data = high_tides, aes(x = Datetime, y = high_tide), color = 'darkgreen') +
  geom_line(data = high_tides, aes(x = Datetime, y = ma_high_tide), color = 'darkgreen', size = 1) +
  geom_smooth(data = monsoon_spring_highs, aes(x = Datetime, y = ma_monsoon_spring_highs), color = 'purple', size = 1) +
  scale_x_datetime(expand = c(0, 0), limits = c(data$Datetime[1], tail(data$Datetime, n=1))) +
  labs(x = 'Year', y = 'Elevation (m)', title = model)
