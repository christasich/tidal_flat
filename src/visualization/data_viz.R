library(feather)
library(tsibble)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df_base = read_feather('../../data/interim/base')
df_gs_0.1 = read_feather('../../data/interim/gs_0.1')
df_rho_1300 = read_feather('../../data/interim/rho_1300')
df_slr_0.005 = read_feather('../../data/interim/slr_0.005')
df_slr_0.008 = read_feather('../../data/interim/slr_0.008')
df_slr_0.01 = read_feather('../../data/interim/slr_0.01')
df_slr_0.02 = read_feather('../../data/interim/slr_0.02')
df_slr_1 = read_feather('../../data/interim/slr_1')
df_ssc_half = read_feather('../../data/interim/ssc_half')
df_ssc_double = read_feather('../../data/interim/ssc_double')
df_longrun = read_feather('../../data/interim/longrun')
test = read_feather('../../test')

index = test$index

ggplot() +
  geom_line(aes(x = index, y = z), data = df_gs_0.1, color = 'blue') +
  geom_line(aes(x = index, y = z), data = df_rho_1300, color = 'red') +
  geom_line(aes(x = index, y = z), data = df_base) 

ggplot(data = df_longrun) +
  geom_line(aes(x = index, y = z), color = 'red') +
  geom_line(aes(x = index, y = h), color = 'blue', alpha = 0.2)

ggplot(data = df_ssc_half) +
  geom_line(aes(x = index, y = z), color = 'red') +
  geom_line(aes(x = index, y = h), color = 'blue', alpha = 0.2)
