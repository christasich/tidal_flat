library(feather)
library(ggplot2)

file = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(file)
setwd('..')

results_path = file.path(getwd(), 'data', 'interim', 'results')

results_path = 'C:\\Users\\chris\\Downloads\\model_runs'

files = list.files(results_path, full.names = TRUE)

data = read_feather(files[3])

ggplot() +
  geom_line(data = data, aes(x = seq(1, length(z)), y = z))