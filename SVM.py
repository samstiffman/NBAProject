#import normalizing
from sklearn import preprocessing
# Pandas for data frames
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import MNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
path="C:\\Users\\samst\\Desktop\\MLProject\\MLProjectMod3\\"\
# Latex output file
LATEXPATH = path + "\\Viz\\SVMConfMats.txt"
LATEXFILE = open(LATEXPATH, "w")
LATEXFILE.close()
    
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

## Plotting SVM From: https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    encoder = preprocessing.OrdinalEncoder()
    Z = encoder.fit_transform(Z.reshape(-1,1))
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def greaterThanFifty(x):
    if x == ' <=50K':
        return 1
    else:
        return -1
##################################################



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

########## TOODOO #####
### Create function that automatically drops all categorical variables


###### Normalization
# Then normalization
# Both standard and min max will be tested
minMaxScaler   = preprocessing.MinMaxScaler()
standardScaled = preprocessing.StandardScaler() 
colLabs = trainData.columns

minMaxTrain = minMaxScaler.fit_transform(trainData)
minMaxTest  = minMaxScaler.fit_transform(testData)

standardTrain    = standardScaled.fit_transform(trainData)
standardTest     = standardScaled.fit_transform(testData)

minMaxSVMDataTrain = pd.DataFrame(minMaxTrain, columns=colLabs)
minMaxSVMDataTest  = pd.DataFrame(minMaxTest, columns=colLabs)

standardSVMTrain    = pd.DataFrame(standardTrain, columns=colLabs)
standardSVMTest     = pd.DataFrame(standardTest, columns=colLabs)


########### SVM first with min-max

SVMMinMaxModel=LinearSVC(C=1)
model = SVMMinMaxModel.fit(minMaxSVMDataTrain, trainLabels)

########## Plotting




fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of SVM ')
# Set-up grid for plotting.
X0, X1 = trainData["Hours-Per-Week"], trainData["Age"]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# Map labels to -1 and 1 so they can be used
numLabels = list(map(greaterThanFifty, trainLabels))
ax.scatter(X0, X1, c=numLabels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Age')
ax.set_xlabel('Hours Per Week')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
#############

####### Training Accuracy
prediction = SVMMinMaxModel.predict(minMaxSVMDataTrain)

print("\nThe prediction from SVM is:")
print(prediction)
print("\nThe actual labels are:")
print(trainLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(trainLabels, prediction)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "SVM MinMax Training Confusion Matrix"))
LATEXFILE.close()

print("\nTraining Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))

####### Testing Accuracy

prediction = SVMMinMaxModel.predict(minMaxSVMDataTest)

print("\nThe prediction from SVM is:")
print(prediction)
print("\nThe actual labels are:")
print(testLabels)

confMat = confusion_matrix(testLabels, prediction)
print("\nTest Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))
#### CONFUSION MATRIX TEXT EXPORT TO LATEX

LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "SVM MinMax Test Confusion Matrix"))
LATEXFILE.close()


############ SVM Standard scaling
SVMStandardModel=LinearSVC(C=1)
model = SVMStandardModel.fit(standardSVMTrain, trainLabels)


########## Plotting


#############

####### Training Accuracy
prediction = SVMMinMaxModel.predict(standardSVMTrain)

print("\nThe prediction from SVM is:")
print(prediction)
print("\nThe actual labels are:")
print(trainLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
confMat = confusion_matrix(trainLabels, prediction)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX

LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "SVM Standard Training Confusion Matrix"))
LATEXFILE.close()

print("\nTraining Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))

####### Testing Accuracy

prediction = SVMStandardModel.predict(standardSVMTest)

print("\nThe prediction from SVM is:")
print(prediction)
print("\nThe actual labels are:")
print(testLabels)

confMat = confusion_matrix(testLabels, prediction)
print("\nTest Accuracy:")
print((confMat[0][0]+confMat[1][1])/(confMat[0][0]+confMat[1][1]+confMat[1][0]+confMat[0][1]))
#### CONFUSION MATRIX TEXT EXPORT TO LATEX

LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confMat, [">50k", "<=50k"],[">50k", "<=50k"], "SVM Standard Test Confusion Matrix"))
LATEXFILE.close()