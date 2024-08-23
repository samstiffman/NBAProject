#import normalizing
from sklearn import preprocessing
# Pandas for data frames
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import MNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn import metrics

############ Utility Functions ##############


### Function to output string version of row 
# Row is row that should be tablefied
# Length should be length of row not counting name
def makeRow(row, length, rowName):
    finalRow = "$" + rowName + "$"
    for i in range(0,length):
            finalRow += "&$" + str(row[i])+ "$"
    finalRow += "\\\\"
    return finalRow
# Function to give confusion matrix as latex table
# Data should be a 2 dimensional symmetric matrix
# Col names should be the names of the columns
# Row names should be the names of the rows in first column
def prettyLatexTables(data, colNames, rowNames, tableName):
    assert len(colNames) == len(rowNames), "colNames and rowNames should have same number of elements"
    size = len(colNames)
    numCols = ["c|" for col in colNames]
    numCols.append("c")
    table = ("\\begin{table}\\begin{tabular}{"+"".join(numCols)+"}" +
             makeRow(colNames, size, "True Value, Output of Model") +
             "\\hline" +
             "".join([makeRow(data[i], size, rowNames[i]) for i in range(0,size)])+
        "\\end{tabular}\n\centering\caption{"+tableName+"}\n"+
        "\end{table}")
    return table



##################################################

path=""
# Latex output file
LATEXPATH = path + "\\Viz\\BayesConfMats.txt"
LATEXFILE = open(LATEXPATH, "w")
LATEXFILE.close()

# load in dataset as pandas dataframe
testData = pd.read_csv(path+"testingData.csv")
trainData = pd.read_csv(path+"trainingData.csv")

testLabels = testData["Label"]
trainLabels = trainData["Label"]

# Remove Labels
testData = testData.drop("Label",axis=1)
trainData = trainData.drop("Label",axis=1)
# First column is a primary key
testData = testData.drop("Unnamed: 0",axis=1)
trainData = trainData.drop("Unnamed: 0",axis=1)

# we need to remove all categorical variables
testData  = testData.drop(columns = ["WorkClass", "Education", "MaritalStatus", "Occupation", "Occupation", "RelationshipToFamily", "Race", "Sex"])
trainData = trainData.drop(columns = ["WorkClass", "Education", "MaritalStatus", "Occupation", "Occupation", "RelationshipToFamily", "Race", "Sex"]) 




###### Normalization
# Then normalization
# ## Normalize the data
#Min-Max scaler used since standard scaler gives negative values which wont work for multinomial naive bayes
minMaxScaler   = preprocessing.MinMaxScaler()
standardScaled = preprocessing.StandardScaler() 
colLabs = trainData.columns

# Standard scaler for naive Bayes
multinomialTrainDataS = minMaxScaler.fit_transform(trainData)
multinomialTestDataS  = minMaxScaler.fit_transform(testData)

gaussianTrainDataS    = standardScaled.fit_transform(trainData)
gaussianTestDataS     = standardScaled.fit_transform(testData)

multinomialTrainData = pd.DataFrame(multinomialTrainDataS, columns=colLabs)
multinomialTestData  = pd.DataFrame(multinomialTestDataS, columns=colLabs)

gaussianTrainData    = pd.DataFrame(gaussianTrainDataS, columns=colLabs)
gaussianTestData     = pd.DataFrame(gaussianTestDataS, columns=colLabs)


# Run both Gaussian and Multinomial Naive Bayes

####### MultiNomial Naive Bayes ##########
multBayes = MultinomialNB(alpha=2)

## Run the model using fit
multBayes.fit(multinomialTrainData, trainLabels)

####### Training Accuracy
predictionOfTest = multBayes.predict(multinomialTrainData)

print("\nThe prediction from NB is:")
print(predictionOfTest)
print("\nThe actual labels are:")
print(trainLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(trainLabels, predictionOfTest)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "Multinomial Bayes Training Confusion Matrix"))
LATEXFILE.close()



### prediction probabilities
## columns are the labels in alphabetical order
## The decimal in the matrix are the prob of being
## that label
print("\nPrediction Probabilities:\n")
print(np.round(multBayes.predict_proba(multinomialTestData),2))
print("\nTraining Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))

####### Testing Accuracy

predictionOfTest = multBayes.predict(multinomialTestData)

print("\nThe prediction from NB is:")
print(predictionOfTest)
print("\nThe actual labels are:")
print(testLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(testLabels, predictionOfTest)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"],"Multinomial Bayes Test Confusion Matrix"))
LATEXFILE.close()

print("\nTest Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))
### prediction probabilities
## columns are the labels in alphabetical order
## The decimal in the matrix are the prob of being
## that label

####### Gaussian Naive Bayes ##########
GaussianBayes = GaussianNB()

## Run the model using fit
GaussianBayes.fit(gaussianTrainData, trainLabels)

####### Training Accuracy
predictionOfTest = GaussianBayes.predict(gaussianTrainData)

print("\nThe prediction from NB is:")
print(predictionOfTest)
print("\nThe actual labels are:")
print(testLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(trainLabels, predictionOfTest)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "Gaussian Bayes Training Confusion Matrix"))
LATEXFILE.close()


### prediction probabilities
## columns are the labels in alphabetical order
## The decimal in the matrix are the prob of being
## that label
print("\nPrediction Probabilities:\n")
print(np.round(GaussianBayes.predict_proba(gaussianTestData),2))
print("\nTraining Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))
####### Testing Accuracy

predictionOfTest = GaussianBayes.predict(gaussianTestData)

print("\nThe prediction from NB is:")
print(predictionOfTest)
print("\nThe actual labels are:")
print(testLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(testLabels, predictionOfTest)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "Gaussian Bayes Test Confusion Matrix"))
LATEXFILE.close()


print("\nTest Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))


################# LATEX CONFUSION MATRIX











