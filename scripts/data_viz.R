library(feather)
library(ggplot2)
library(tidyverse)

file = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(file)
setwd('..')

results = read.table('./scripts/output.csv', header = TRUE, sep = ',') %>%
  filter(sscfactor <= 3 & sscfactor >= 0 & slr <=0.01)

ggplot(results, aes(x=slr, y=sscfactor, z=inundation.hours, fill=inundation.hours)) +
  geom_raster(interpolate = F, na.rm = TRUE) +
  scale_fill_viridis_c(option='plasma') +
  scale_y_log10() +
  geom_contour(bins=10, color='white')
