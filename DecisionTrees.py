# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:20:37 2022

@author: samst
"""

#import normalizing
from sklearn import preprocessing
# Pandas for data frames
import pandas as pd
import numpy as np
# Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# VIZ
import matplotlib.pyplot as plt
import graphviz 
from sklearn.metrics import confusion_matrix

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


## Prep Work

path="C:\\Users\\samst\\Desktop\\MLProject\\MLProjectMod3\\"
# Latex output file
LATEXPATH = path + "\\Viz\\DTConfMats.txt"
LATEXFILE = open(LATEXPATH, "w")
LATEXFILE.close()

# load in dataset as pandas dataframe
trainData = pd.read_csv(path+"trainingData.csv")
testData  = pd.read_csv(path+"testingData.csv")

testLabels = testData["Label"]
trainLabels = trainData["Label"]

# Remove Labels
testData = testData.drop("Label",axis=1)
trainData = trainData.drop("Label",axis=1)
# First column is a primary key
testData = testData.drop("Unnamed: 0",axis=1)
trainData = trainData.drop("Unnamed: 0",axis=1)

# Cast non-numerical data types to numerical,
# This works for decision tree but not naive bayes
# Categorical WorkClass - Sex


encoder = preprocessing.OrdinalEncoder()

testData["WorkClass"] = encoder.fit_transform(np.array(testData["WorkClass"]).reshape(-1,1))
testData["Education"] = encoder.fit_transform(np.array(testData["Education"]).reshape(-1,1))
testData["MaritalStatus"] = encoder.fit_transform(np.array(testData["MaritalStatus"]).reshape(-1,1))
testData["Occupation"] = encoder.fit_transform(np.array(testData["Occupation"]).reshape(-1,1))
testData["RelationshipToFamily"] = encoder.fit_transform(np.array(testData["RelationshipToFamily"]).reshape(-1,1))
testData["Race"] = encoder.fit_transform(np.array(testData["Race"]).reshape(-1,1))
testData["Sex"] = encoder.fit_transform(np.array(testData["Sex"]).reshape(-1,1))


trainData["WorkClass"] = encoder.fit_transform(np.array(trainData["WorkClass"]).reshape(-1,1))
trainData["Education"] = encoder.fit_transform(np.array(trainData["Education"]).reshape(-1,1))
trainData["MaritalStatus"] = encoder.fit_transform(np.array(trainData["MaritalStatus"]).reshape(-1,1))
trainData["Occupation"] = encoder.fit_transform(np.array(trainData["Occupation"]).reshape(-1,1))
trainData["RelationshipToFamily"] = encoder.fit_transform(np.array(trainData["RelationshipToFamily"]).reshape(-1,1))
trainData["Race"] = encoder.fit_transform(np.array(trainData["Race"]).reshape(-1,1))
trainData["Sex"] = encoder.fit_transform(np.array(trainData["Sex"]).reshape(-1,1))
#############    Decision Tree   #######################

# Set up decision Tree
# Worked slightly better with gini impurity than entropy
decisionTree=DecisionTreeClassifier(criterion='gini', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=20, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.001, # Set to reduce tree to manageable size
                            class_weight=None)

# Build model from training data
decisionTree.fit(trainData, trainLabels)
# Plot Tree
#tree.plot_tree(decisionTree)
featureNames=trainData.columns
dot_data = tree.export_graphviz(decisionTree, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=featureNames,  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
graph = graphviz.Source(dot_data) 
graph.render('Decision Tree') 


print("\nTrain Accuracy:")
prediction=decisionTree.predict(trainData)
confusionMatrix = confusion_matrix(trainLabels, prediction)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confusionMatrix, [">50k", "<=50k"],[">50k", "<=50k"], "Decision Tree Training Confusion Matrix"))
LATEXFILE.close()

accuracy = (confusionMatrix[0][0]+confusionMatrix[1][1])/(confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0]+confusionMatrix[1][1])
print(accuracy)
## Eval tree on test set


#print("Prediction\n")
prediction=decisionTree.predict(testData)
#print(prediction)
## Show the confusion matrix
confusionMatrix = confusion_matrix(testLabels, prediction)
print("\nThe confusion matrix is:")
print(confusionMatrix)
#### CONFUSION MATRIX TEXT EXPORT TO LATEX
LATEXFILE = open(LATEXPATH, "a")
LATEXFILE.write(prettyLatexTables(confusionMatrix, [">50k", "<=50k"],[">50k", "<=50k"], "Decision Tree Test Confusion Matrix"))
LATEXFILE.close()


print("\nTest Accuracy:")
accuracy = (confusionMatrix[0][0]+confusionMatrix[1][1])/(confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0]+confusionMatrix[1][1])
print(accuracy)
