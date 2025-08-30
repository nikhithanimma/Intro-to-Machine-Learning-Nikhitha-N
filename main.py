import pandas as pd
import GWCutilities as util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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
X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 6)

##Defining both of the models
dt_model = DecisionTreeClassifier(max_depth=6, class_weight="balanced")
rf_model = RandomForestClassifier(n_estimators=5,class_weight="balanced", max_depth=2)


dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

def evaluate_model(model, model_name, show_tree=False):
    """Prints accuracy, confusion matrix, and optional tree plots."""

    print(f"\n---{model_name} Results ---")
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    print("Test Accuracy:", accuracy_score(y_test, test_pred))
    print("Train Accuracy:", accuracy_score(y_train, train_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred, labels = [1,0]))

    if show_tree and model_name == "Decision Tree":
        util.printTree(model, X.columns)
    elif show_tree and model_name == "Random Forest":
        for i, tree in enumerate(model.estimators_):
            plot_tree(tree, feature_names=X.columns, filled = True)
            plt.title(f"Random Forest Tree {i+1}")
            plt.show()
            


def user_input_prediction(): 
    print("\nEnter the following information to predict Heart Disease:\n")

    sex = input("Sex (Male/Female): ")
    age = input("AgeCategory (e.g. 18-24, 25-29,...): ")
    bmi = float(input("BMI (number): "))
    smoking = input ("Smoking (Yes/No): ")
    alcohol = input("AlchoholDrinking (Yes/No): ")
    activity = input("PhysicalActivity (Yes/No): ")
    health = input("GenHealth (Poor, Fair, Good, Very good, Excellent): ")
    race = input("Race (White, Black, Asian, Hispanic, Other): ")

    user_df = pd.DataFrame([{
        "BMI": bmi,
        "Sex": sex, 
        "AgeCategory": age, 
        "Smoking": smoking,
        "AlcoholDrinking": alcohol, 
        "PhysicalActivity": activity, 
        "GenHealth": health, 
        "Race": race
    }])

    user_df = util.labelEncoder(user_df, [
        "Smoking", "AlcoholDrinking", "Sex", 
        "AgeCategory", "PhysicalActivity", "GenHealth"
    ])
    user_df = util.oneHotEncoder(user_df, ["Race"])

    user_df = user_df.reindex(columns=X.columns,fill_value=0 )

    dt_pred = dt_model.predict(user_df)[0]
    rf_pred = rf_model.predict(user_df)[0]

    print("\n--- Prediction Comparison ---")
    print("Decision Tree says:", "Heart Disease" if dt_pred == 1 else "No Heart Disease")
    print("Random Forest says:", "Heart Disease" if rf_pred == 1 else "No Heart Disease")

#Interactive flow of the chat
choice = input("\nWhich method would you like to see first? (Decision/Random): ").strip().lower()

if choice.startswith("d"):
    evaluate_model(dt_model, "Decision Tree", show_tree = True)
    other = input("\n Would like to also see the results of the Random Forest? (yes/no): ").strip().lower()
    if other.startswith("y"):
        evaluate_model(rf_model, "Random Forest",
        show_tree=True)
else: 
    evaluate_model(rf_model, "Random Forest",
show_tree=True)
    other = input("\n Would like to also see the results of the Decision Tree? (yes/no): ").strip().lower()
    if other.startswith("y"):
        evaluate_model(dt_model, "Decision Tree",
        show_tree=True)

user_input_prediction()