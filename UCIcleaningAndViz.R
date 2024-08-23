# load dplyr


install.packages('epiDisplay')
library(epiDisplay)
# adult_data imported manually into R
# Used to see frequency for each categorical variable
# Note that the images will not load if enough space is not given to the plot window
par(mfrow = c(2,2), mar = c(1,2,2,1))
tab1(adult_data$'V2', sort.group='decreasing', main='Employement Sector Frequency')
tab1(adult_data$'V4', sort.group='decreasing', main='Education Frequency')
tab1(adult_data$'V6', sort.group='decreasing', main='Marriage Frequency')
tab1(adult_data$'V7', sort.group='decreasing', main='Job Type Frequency')
par(mfrow = c(2,2), mar = c(2,1,1,2))
tab1(adult_data$'V8', sort.group='decreasing', main='Family Frequency')
tab1(adult_data$'V9', sort.group='decreasing', main='Race Frequency')
tab1(adult_data$'V10', sort.group='decreasing', main='Sex Frequency')
par(mfrow = c(1,1), mar = c(1,1,1,1))
tab1(adult_data$'V14'[adult_data$'V14' != "Outlying-US(Guam-USVI-etc)"], sort.group='decreasing', main='Nationality Frequency')

#Viz showing capital gains and capital losses data is too heavily skewed to be of use
par(mfrow = c(2,1), mar = c(2,2,2,2))
hist(adult_data$'V11',xlab='capital_gains', main ='Histogram of Capital Gains')
hist(adult_data$'V12',xlab='capital_losses', main ='Histogram of Capital Losses')

# Remove the country data, and capital gain and loss cols since they are absurdly heavily skewed
cleaned <- adult_data[c(1:10,13,15)]
# Remove rows with missing values since the appearance is low compared to the 32000 entries
# First change ? to NA
cleaned[cleaned == ' ?'] <- NA

#Next use na.omit
cleaned <- na.omit(cleaned)

#Redraw frequency tables of remaining variables to see effect of removing ?'s
tab1(cleaned$'V2', sort.group='decreasing', main='Employement Sector Frequency')
tab1(cleaned$'V4', sort.group='decreasing', main='Education Frequency')
tab1(cleaned$'V6', sort.group='decreasing', main='Marriage Frequency')
tab1(cleaned$'V7', sort.group='decreasing', main='Job Type Frequency')
tab1(cleaned$'V8', sort.group='decreasing', main='Family Frequency')
tab1(cleaned$'V9', sort.group='decreasing', main='Race Frequency')
tab1(cleaned$'V10', sort.group='decreasing', main='Sex Frequency')


# Now that we have a handle on categorical data lets viz and clean the quant data
# to see distribution of age we can just use a histogram this dist could be important for analyses
# Distribution looks like we might expect from a population not including children
hist(cleaned$'V1', main='Distribution of Age')
# 5 number summary to spot outliers
summary(cleaned$'V1')
# Summary gave max as 90 which is not unusual

# Now for fnlgwt
hist(cleaned$'V3', main='Distribution of fnglwt')
boxplot(cleaned$'V3', main='Boxplot of fnglwt')
# This variable was used in the paper the dataset was used in

#Education number is just a duplicate of education so we can safely remove it
cleaned <- cleaned[-5]


# Finally write the cleaned data to CSV Format
write.csv(cleaned,"C:\\Users\\samst\\Desktop\\MLProject\\cleanedUCIData.csv")
