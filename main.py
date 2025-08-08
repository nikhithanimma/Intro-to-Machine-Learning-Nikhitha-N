import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset


print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows


input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset


print("\nHere is a preview of the dataset after label encoding. \n")


input("\nPress Enter to continue.\n")

#One hot encode the dataset


print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)


input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split





from sklearn.tree import DecisionTreeClassifier






#Test the model with the testing data set and prints accuracy score


from sklearn.metrics import accuracy_score




#Prints the confusion matrix
from sklearn.metrics import confusion_matrix





#Test the model with the training data set and prints accuracy score




input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")





#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")