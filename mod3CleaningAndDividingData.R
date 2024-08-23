#setwd("")
supervisedData <- read.csv("cleanedUCIData.csv", header=TRUE, row.names=1)

# Remove fnlwgt since we dont know what it really is
supervisedData <- supervisedData[-3]

# Make sure structure of data is correct
str(supervisedData)
# Convert Char to Factor
supervisedData$V2  <- as.factor(supervisedData$V2)
supervisedData$V4  <- as.factor(supervisedData$V4)
supervisedData$V6  <- as.factor(supervisedData$V6)
supervisedData$V7  <- as.factor(supervisedData$V7)
supervisedData$V8  <- as.factor(supervisedData$V8)
supervisedData$V9  <- as.factor(supervisedData$V9)
supervisedData$V10 <- as.factor(supervisedData$V10)
supervisedData$V15 <- as.factor(supervisedData$V15)
# We need balanced labels for decision tree to work
table(supervisedData$V15)
# As we can see there are 3* as many people with <= 50k as there are >50k so we need to somehow take a balanced sample for training
# For testing it doesn't actually need to be balanced.
# To do this we undersample
lessThan50    <- which(supervisedData$V15 == ' <=50K')
greaterThan50 <- which(supervisedData$V15 == ' >50K')

# Since we had 7000 >50k points lets go for 10k sample size 5k of each 
n <- 10000

# Before we sample lets rename the columns so we arent dealing with the annoying V's that mean nothing
names(supervisedData) <- c('Age', 'WorkClass', 'Education', 'MaritalStatus', 'Occupation', 'RelationshipToFamily', 'Race', 'Sex', 'Hours-Per-Week', 'Label')

# Undersampling
lessThan50Sample    <- sample(lessThan50, floor(n/2))
greaterThan50Sample <- sample(greaterThan50, floor(n/2))

# Set seed so samples are reproducible
set.seed(22)
sample = c(lessThan50Sample, greaterThan50Sample)

# Disjoint test and training data
trainData <- supervisedData[sample, ]
testData  <- supervisedData[-sample, ]
table(trainData$Label)
# Training data is now perfectly balanced, test data need not be balanced
table(testData$Label)

### Export train and test data to CSV so they can be used in the python code
# This can be used for all the supervised methods
write.csv(trainData,"trainingData.csv")
write.csv(testData,"testingData.csv")
write.csv(supervisedData,"fullData.csv")
