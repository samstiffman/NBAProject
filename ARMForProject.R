#Import Data set
setwd("C:\\Users\\samst\\Desktop\\\\MLProject\\MLProjectMod2")
cleanedCensusData <- read.csv("cleanedCensusData.csv", header=FALSE)
#Delete Header
cleanedCensusData <- cleanedCensusData[-1, ]

# Prep for ARM
# First we need to remove all label data since ARM is unsupervised

censusForArmCopy <- cleanedCensusData[-1]


#install.packages("arules")
#library(arules)

# HS dropout and HS grad rate should be very strongly inversely related
censusForArmCopy <- censusForArmCopy[-1]

# First we need to format the data as transaction data
# To do this each column will be discretized
# This will first be done in a systematic way according to
# The 25th, 50th, and 75th percentiles
# Values in the range 0-25 will be low with symbol L 
# Values in the range 25-50 will be medlow with symbol ML
# Values in the range 50-75 will be medhigh with symbol MH
# Values in the range 75-100 will be high with symbol H

# Percentile Calculation
# First we need to make a copy so the logical vectors are always valid
censusForARM <- censusForArmCopy
for(i in 1:15){
  # Calculate percentiles for given column
  percentile <- quantile(as.double(censusForArmCopy[,i]), probs = c(.25, .5, .75))
  #Classify values in Column
  if (i<= 5) { # Education Cats
    censusForARM[as.double(censusForArmCopy[,i]) <= percentile[2],i] <- paste("MLE", i)
    censusForARM[as.double(censusForArmCopy[,i]) >  percentile[2],i] <- paste("MHE", i)
    # We can actually reduce the complexity of the logical expressions by overwritting the tails
    censusForARM[as.double(censusForArmCopy[,i]) <= percentile[1],i] <- paste("LE", i)
    censusForARM[as.double(censusForArmCopy[,i]) >  percentile[3],i] <- paste("HE", i)
  }
  else{ # Income
    censusForARM[as.double(censusForArmCopy[,i]) <= percentile[2],i] <- paste("MLI", i-5)
    censusForARM[as.double(censusForArmCopy[,i]) >  percentile[2],i] <- paste("MHI", i-5)
    # We can actually reduce the complexity of the logical expressions by overwritting the tails
    censusForARM[as.double(censusForArmCopy[,i]) <= percentile[1],i] <- paste("LI", i-5)
    censusForARM[as.double(censusForArmCopy[,i]) >  percentile[3],i] <- paste("HI", i-5)
  }
  
}

# We can save the CensusTransactionData 
write.csv(censusForARM,"censusTransactionData.csv")


library(arules)
# Now its time to do the actual Association Rule Mining
censusTransactions <- read.transactions("censusTransactionData.csv",
                                rm.duplicates = FALSE, 
                                format = "basket",  ##if you use "single" also use cols=c(1,2)
                                sep=",",  ## csv file
                                cols=1) ## The dataset HAS row numbers
# apriori can generate the rules
# We need to decide on the thresholds
# 
#  The data has 15 categories in each row 
# This will generate a frequency plot for each column with frequencies of the 4 character strings in that column
# These plots helped diagnose some serious bugs that were throwing off results
library(epiDisplay)
for( i in 1:15){
  tab1(censusForARM[,i], sort.group='decreasing', main=paste('V',i+1,sep=""))
}
# Looking at these plots we can verify that the code works since each category is approximately uniformly distributed

# vector so that all rules cross from education to income
educationVars = c('LE 1', 'LE 2', 'LE 3', 'LE 4', 'LE 5', 'MLE 1', 'MLE 2', 'MLE 3', 'MLE 4', 'MLE 5', 'MHE 1', 'MHE 2', 'MHE 3', 'MHE 4', 'MHE 5', 'HE 1', 'HE 2', 'HE 3', 'HE 4', 'HE 5')
incomeVars    = c('LI 1', 'LI 2', 'LI 3', 'LI 4', 'LI 5', 'MLI 1', 'MLI 2', 'MLI 3', 'MLI 4', 'MLI 5', 'MHI 1', 'MHI 2', 'MHI 3', 'MHI 4', 'MHI 5', 'HI 1', 'HI 2', 'HI 3', 'HI 4', 'HI 5', 'LI 6', 'LI 7', 'LI 8', 'LI 9', 'LI 10', 'MLI 6', 'MLI 7', 'MLI 8', 'MLI 9', 'MLI 10', 'MHI 6', 'MHI 7', 'MHI 8', 'MHI 9', 'MHI 10', 'HI 6', 'HI 7', 'HI 8', 'HI 9', 'HI 10')

APPEARANCEVEC = list(lhs = educationVars, rhs = incomeVars)
# Run Apriori
censusRules = arules::apriori(censusTransactions, parameter = list(support=.05, minlen=2), appearance = APPEARANCEVEC)
sortedRules <- arules::sort(censusRules, by="lift", decreasing=TRUE)
inspect(sortedRules)

# This implies the variables are not independent uniform which is good as we expect some of the columns to be highly correlated

# Lift >1

# We can look at most confident, most supported and most lifted rules

inspect(head(sortedRules, 15, by="support"))
inspect(head(sortedRules, 15, by="confidence"))
inspect(head(sortedRules, 15, by="lift"))
# we can see every rule has lift>1 

# Leftover columns should be in order:
# HS Grad Rate, Grad/Prof rate, <10k rate, >200k rate

### Viz time
#install.packages("TSP")
#install.packages("data.table")
#install.packages("arulesViz", dependencies = TRUE)
library(TSP)
library(data.table)
library(arulesViz)

# Graph of rules
# Only take the most confident rules
sortedRules <- arules::sort(censusRules, by="confidence", decreasing=TRUE)
plot(sortedRules, method = "graph",engine = "interactive", limit = 20)
plot(sortedRules, method="graph", engine="htmlwidget", limit = 20)
