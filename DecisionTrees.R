#install.packages('rpart')
#install.packages('rattle')

#Decision Trees
library(rpart)
library(rattle)  ## FOR Decision Tree Vis
setwd("C:\\Users\\samst\\Desktop\\MLProject\\MlPRojectMod3")
decisionTreeData <- read.csv("fullData.csv", header=TRUE, row.names=1)

testData  <- read.csv("testingData.csv", header=TRUE, row.names=1)
trainData <- read.csv("trainingData.csv", header=TRUE, row.names=1)

##### Unfortunately R always loads factors as char
decisionTreeData$Label                 <- as.factor(decisionTreeData$Label)
decisionTreeData$WorkClass             <- as.factor(decisionTreeData$WorkClass)
decisionTreeData$Education             <- as.factor(decisionTreeData$Education)
decisionTreeData$MaritalStatus         <- as.factor(decisionTreeData$MaritalStatus)
decisionTreeData$Occupation            <- as.factor(decisionTreeData$Occupation)
decisionTreeData$RelationshipToFamily  <- as.factor(decisionTreeData$RelationshipToFamily)
decisionTreeData$Race                  <- as.factor(decisionTreeData$Race)
decisionTreeData$Sex                   <- as.factor(decisionTreeData$Sex)


trainData$WorkClass            <- as.factor(trainData$WorkClass)
trainData$Education            <- as.factor(trainData$Education)
trainData$MaritalStatus        <- as.factor(trainData$MaritalStatus)
trainData$Occupation           <- as.factor(trainData$Occupation)
trainData$RelationshipToFamily <- as.factor(trainData$RelationshipToFamily)
trainData$Race                 <- as.factor(trainData$Race)
trainData$Sex                  <- as.factor(trainData$Sex)
trainData$Label                <- as.factor(trainData$Label)


testData$WorkClass             <- as.factor(testData$WorkClass)
testData$Education             <- as.factor(testData$Education)
testData$MaritalStatus         <- as.factor(testData$MaritalStatus)
testData$Occupation            <- as.factor(testData$Occupation)
testData$RelationshipToFamily  <- as.factor(testData$RelationshipToFamily)
testData$Race                  <- as.factor(testData$Race)
testData$Sex                   <- as.factor(testData$Sex)
testData$Label                 <- as.factor(testData$Label)

# Notice no char types only factor which is good
str(testData)
str(trainData)
##################


# Set the seed so the trees are reproducible
set.seed(20)
######## First Tree using mixed data ##################

# Remove Labels from test data
testLabels <- testData$Label
testData <- testData[1:9]

# Convert label to factor
trainData$Label <-as.factor(trainData$Label)

# Run rpart decision tree
# Uses entropy as opposed to Gini
tree <- rpart(trainData$Label ~ ., data = trainData, method="class", parms = list(split="information"))
summary(tree)

prediction= predict(tree, testData, type="class")
## Confusion Matrix
table(prediction,testLabels) ## one way to make a confusion matrix

# Compute Accuracies

(confMat <- as.data.frame(table(prediction,testLabels)))
"Test Accuracy"
(accuracy <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))

"Train Accuracy"
prediction= predict(tree, trainData, type="class")
(confMat <- as.data.frame(table(prediction,trainData$Label)))
(accuracy <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))

## VIS..................
fancyRpartPlot(tree)

#######################################################
######## Second Tree using discrete Data ##############
#######################################################
# We want to discretize:
# V1 = Age
# V13 = hours-per-week

# Age will be simple
# <=18  is a child
# 19-40 is adult
# 40-60 is middle aged
# 60+   is old
decisCopy <- decisionTreeData
decisionTreeData$'Age'[as.double(decisCopy$'Age') <   40] <- 'Adult'
decisionTreeData$'Age'[as.double(decisCopy$'Age') <=  18] <- 'Child'
decisionTreeData$'Age'[as.double(decisCopy$'Age') >=  40] <- 'Middle Aged'
decisionTreeData$'Age'[as.double(decisCopy$'Age') >=  60] <- 'Old'

# Hours worked will be based slightly on the legalities
# 35 - 50  is full time
# <35 is part time
# >50 is absurd
decisionTreeData$'Hours.Per.Week'[as.double(decisCopy$'Hours.Per.Week') <   50] <- 'Full Time'
decisionTreeData$'Hours.Per.Week'[as.double(decisCopy$'Hours.Per.Week') <   35] <- 'Part Time'
decisionTreeData$'Hours.Per.Week'[as.double(decisCopy$'Hours.Per.Week') >=  50] <- 'Absurd'

############# SAMPLING ###############
# We need to resample
# As we can see there are 3* as many people with <= 50k as there are >50k so we need to somehow take a balanced sample for training
# For testing it doesn't actually need to be balanced.
# To do this we undersample
lessThan50    <- which(decisionTreeData$'Label' == ' <=50K')
greaterThan50 <- which(decisionTreeData$'Label' == ' >50K')

# Since we had 7000 >50k points lets go for 10k sample size 5k of each 
n <- 10000
# Undersampling
lessThan50Sample    <- sample(lessThan50, floor(n/2))
greaterThan50Sample <- sample(greaterThan50, floor(n/2))
sample              <- c(lessThan50Sample, greaterThan50Sample)

# Disjoint test and training data
trainData <- decisionTreeData[sample, ]
testData  <- decisionTreeData[-sample, ]

# Remove Labels from test data
testLabels <- testData$'Label'
testData   <- subset (testData, select = -Label)

# Convert label to factor
trainData$'Label' <- as.factor(trainData$'Label')

################# Decision Tree##############
# Run rpart decision tree
# cp determines size of tree
tree <- rpart(trainData$Label ~ ., data = trainData, method="class", cp=0.012)
summary(tree)

prediction <- predict(tree, testData, type="class")
## Confusion Matrix
table(prediction,testLabels) ## one way to make a confusion matrix

# Compute Accuracy
confMat   <- as.data.frame(table(prediction,testLabels))
"Test Accuracy"
(accuracy <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))

"Train Accuracy"
prediction <- predict(tree, trainData, type="class")
confMat    <- as.data.frame(table(prediction,trainData$'Label'))
(accuracy  <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))


## VIS..................
fancyRpartPlot(tree)