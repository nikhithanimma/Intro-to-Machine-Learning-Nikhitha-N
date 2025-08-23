import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "Smoking", "AlcoholDrinking", "Sex", "AgeCategory", "PhysicalActivity", "GenHealth"])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())

input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 6)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 5, class_weight = 'balanced', max_depth = 2)
clf = clf.fit(X_train, y_train)





#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions)
print("The accuracy with the testing data set of the Desicion Tree is: " + str(test_acc))


#Prints the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions, labels=[1,0])
print("The confusion matrix of the tree is: ")
print(cm)




#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predictions)
print("The accuracy with the training data set of the Desicion Tree is: " + str(train_acc))



input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")
print("\n A Decision Tree can be used in another field such as business in order to reccomend products or predict how the stock market will perform, based on data from the past. \n To ensure the model performs fairly, it is important to avoid biased data, maintain a balanced distribution to make sure the model doesn't facor one group, and avoid overfitting by limiting the max depth.")



#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

for x in range(0, 5):
    plot_tree(clf.estimators_[x], feature_names=X.columns, filled=True)
    plt.show()