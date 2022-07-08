library(lme4) 
library(plyr)


data <- read.csv(file = '../formatted_data/regression_data.csv')
model <- glmer.nb(count ~ -1 + (1+Monday+Tuesday+Wednesday+Thursday+Friday+Saturday+Sunday+February+March+April+May+June|department_name), data=data, control = glmerControl(optimizer ="bobyqa"))
summary(model)