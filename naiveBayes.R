library(naivebayes)
library(e1071)

#setwd("")
naiveBayesData <- read.csv("cleanedUCIData.csv", header=TRUE, row.names=1)

### Since I made training and testing data for Decision Trees we will use those files
testData <- read.csv("testingData.csv", header=TRUE, row.names=1)
trainData <- read.csv("trainingData.csv", header=TRUE, row.names=1)

##### Unfortunately R always loads factors as char
naiveBayesData$V2  <- as.factor(naiveBayesData$V2)
naiveBayesData$V4  <- as.factor(naiveBayesData$V4)
naiveBayesData$V6  <- as.factor(naiveBayesData$V6)
naiveBayesData$V7  <- as.factor(naiveBayesData$V7)
naiveBayesData$V8  <- as.factor(naiveBayesData$V8)
naiveBayesData$V9  <- as.factor(naiveBayesData$V9)
naiveBayesData$V10 <- as.factor(naiveBayesData$V10)
naiveBayesData$V15 <- as.factor(naiveBayesData$V15)
str(naiveBayesData)

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

# Remove Labels from test data
testLabels <- testData$Label
testData   <- testData[1:9]

#### Run Naive Bayes

# Leave laplace smoothing at 1 for now
NaiveBayesModel <- naiveBayes(trainData$Label ~ ., data = trainData, laplace = 1)

prediction <- predict(NaiveBayesModel, testData)

(confMat  <- as.data.frame(table(prediction,testLabels)))
"Test Accuracy"
(accuracy <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))

"Train Accuracy"
prediction <- predict(NaiveBayesModel, trainData)
(confMat   <- as.data.frame(table(prediction,trainData$Label)))
(accuracy  <- (confMat$Freq[1] + confMat$Freq[4])/sum(confMat$Freq))




