

# For putting multiple plots together
install.packages('epiDisplay')
library(epiDisplay)

# census data imported manually into R
# Used to see frequency for each categorical variable


# First notice that the Region column is empty and should contain the information in the Name column
# However since the data contains only Minnesota data these both can be safely removed
cleaned <- rawCensusData[3:20]

# Similarly the state column provides no information and can be safely removed
# The county column is labeled data and contains the exact same information as the names so it also can be removed
cleaned <- cleaned[1:16]

# The final thing that should be done before exploratory viz is to change the col names to a human readable format
schoolVars <- c("HS Dropout rate", "HS grad rate", "Some College rate", "Associates rate", "Bachelors rate", "Grad/Prof rate")
moneyVars <- c(" <10k rate", "10-15k rate", "15-25k rate", "25-35k rate", "35-50k rate", "50-75k rate", "75-100k rate", "100-150k rate", "150-200k rate", ">200k rate")
names(cleaned) <- c(schoolVars, moneyVars)
# Finally write the cleaned data to CSV Format
write.csv(cleaned,"C:\\Users\\samst\\Desktop\\MLProject\\cleanedCensusData.csv")
