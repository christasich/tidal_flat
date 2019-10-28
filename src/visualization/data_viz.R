library(feather)
library(tsibble)

index = seq.POSIXt(pressure$Datetime[1], tail(ssc$Datetime,1), by = "min")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df_base = read_feather('../../data/interim/base')
df_gs_0.1 = read_feather('../../data/interim/gs_0.1')
df_rho_1300 = read_feather('../../data/interim/rho_1300')
df_slr_0.005 = read_feather('../../data/interim/slr_0.005')
df_slr_0.008 = read_feather('../../data/interim/slr_0.008')
df_slr_0.01 = read_feather('../../data/interim/slr_0.01')
test = read_feather('../../test')

ggplot(aes(x = df_base.index)