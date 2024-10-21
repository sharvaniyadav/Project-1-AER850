# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:21:44 2024

@author: Sharvani Yadav
"""
# The objective of this project to predict the step of an inverter based on X, Y, and Z coordinates. 
# The classification problem involves 13 different classes.


# Import necessary libraries
import numpy as np                                                                                 # NumPy is used for numerical operations, mainly arrays and mathematical functions.
import seaborn as sb                                                                              # Seaborn is used for creating advanced visualizations, especially statistical graphics.
import pandas as pd                                                                                # Pandas is a library used for data manipulation and analysis, particularly for working with dataframes.
import matplotlib.pyplot as plt                                                                    # Matplotlib is a plotting library used for creating static, animated, and interactive visualizations.

# Import specific modules from scikit-learn for machine learning tasks
from sklearn.model_selection import train_test_split                                               # This function is used to split your data into training and testing sets.
from sklearn.preprocessing import StandardScaler                                                   # StandardScaler is used for feature scaling, normalizing data to improve model performance.
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression                                                # LogisticRegression is a classification algorithm used for binary/multiclass classification.
from sklearn.ensemble import RandomForestClassifier                                                # RandomForestClassifier is an ensemble learning method based on decision trees.
from sklearn.svm import SVC                                                                        # Support Vector Classifier (SVC) is a classification algorithm based on finding a decision boundary between classes.
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, f1_score      # These metrics help evaluate the performance of classification models.
import joblib                                                                                      # Joblib is used to save and load models, making it easier to persist and reuse trained models.
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.1) STEP 1: Data Processing"

df = pd.read_csv('Project_1_Data.csv') 
df = df.dropna()

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.2) STEP 2: Data Visualization"

"This code creates 3D visualization of the data:"

from mpl_toolkits.mplot3d import Axes3D                                                            # Axes3D is used to create 3D plots for visualizing data in 3 dimensions (X, Y, Z).

# Create a 3D scatter plot to visualize the coordinates (X, Y, Z)
fig = plt.figure()                                                                                 # Create a new figure for plotting.
ax = fig.add_subplot(111, projection='3d')                                                         # Add a 3D subplot to the figure.

# Plot the data points on the 3D plot using X, Y, and Z coordinates from the dataframe
ax.scatter(df['X'], df['Y'], df['Z'])                                                              # Scatter plot with X, Y, Z columns from the dataframe.
ax.view_init(42, 185)                                                                              # Set the view angle for better visibility (30 degrees elevation and 185 degrees azimuth).

# Show the 3D scatter plot
plt.title("3D Visualization of Data")
plt.show()  # Display the plot.

# Count how many times each step appears
step_count = df["Step"].value_counts()

# Draw a bar graph of the step counts
step_count.plot(kind="bar")

# Add labels to explain the graph
plt.title("Distribution of Steps of Inverter")
plt.xlabel("Step")
plt.ylabel("Number of Instances")
plt.show()

"This code creates 2D visualization Bar Graph of the data:"
'''This code analyzes and visualizes the distribution of the Step variable in your dataset. 
It counts how many times each unique step appears, creates a bar graph to display these counts,
and adds labels to explain the graph. The final output helps understand how frequently 
each step occurs, revealing trends or imbalances in data.'''

'''Observations of Graph: 
1. Imbalance in Data: 
The bar graph shows that steps 7, 8, and 9 
have a lot more data points than the other steps. This means the 
model might perform well on these steps but poorly on the others.

2. Prediction Issues: 
If the model is trained on this uneven data, 
it may learn to predict steps 7, 8, and 9 accurately but struggle with the less frequent steps.

3. Solution - StratifiedSampling: 
To fix this, I can use a method called StratifiedShuffleSplit. 
This method divides the data into training and testing sets while 
keeping the same proportion of each step as in the original dataset.

4. Better Performance: 
By using this method, the model will learn from a balanced mix of all steps, 
improving its ability to predict accurately for all steps, not just the ones with more data.'''

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"DATA SPLITTING, TRAINING AND TESTING"

"This code is for splitting the data into training and testing sets:"

my_splitter = StratifiedShuffleSplit(n_splits = 1,                             # Data Splits Only Once    
                                     test_size = 0.2,                          # 20% Data for Testing  
                                     random_state = 42)                        # Self Explanatory :D

for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

"This code is spliting test data vs train data by coordinates & steps:"
coord_train = strat_df_train.drop("Step", axis = 1)
step_train = strat_df_train["Step"]
coord_test = strat_df_test.drop("Step", axis = 1)
step_test = strat_df_test["Step"]


'''Prepping the Dataset:  '''

coord_scaler = StandardScaler()                                                # Create a scaler to standardize the data by removing the average and adjusting the scale.

coord_scaler.fit(coord_train)                                                  # Fit the scaler to the training data so it learns the average and spread of the features.

scaled_data_train = coord_scaler.transform(coord_train)                        # Use the fitted scaler to transform the training data, applying the standardization.

scaled_data_train_df = pd.DataFrame(scaled_data_train,                         # Convert the scaled training data back into a DataFrame to keep the same column names.
                                    columns = coord_train.columns)
coord_train = scaled_data_train_df                                             # Update the original training data variable to now hold the scaled data.

scaled_data_test = coord_scaler.transform(coord_test)                          # Apply the same scaling to the test data using the same parameters learned from the training data.

scaled_data_test_df = pd.DataFrame(scaled_data_test,                           # Convert the scaled test data into a DataFrame, preserving the column names.
                                   columns = coord_test.columns)
coord_test = scaled_data_test_df                                               # Update the original test data variable to now hold the scaled data.

'''The standard deviation of the three features differs significantly, with X being 
more spread out than Z, which means X could have a greater impact on the model.Although the 
values aren't vastly different, this could still influence the model's performance.Therefore,
it’s a good idea to apply scaling methods to improve the model's effectiveness.'''

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.3) STEP 3: Correlation Analysis"

# Visualize the correlation matrix
df.corr()
sb.heatmap(df.corr().round(2), annot=True, cmap="magma")

corr_x = step_train.corr(coord_train['X'])
print(corr_x)

corr_y = step_train.corr(coord_train['Y'])
print(corr_y)

corr_z = step_train.corr(coord_train['Z'])
print(corr_z)

plt.title("Heatmap of Correlation Matrix") 
plt.show()

''' Based on the heatmap, the input variables don’t strongly correlate with 
each other (the highest is around 0.2). This means none of the variables are 
too similar, so none of the data needs to be dropped from the dataset itself.'''

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.4) STEP 4: Classification Model Development/Engineering"

"Model 1 - LOGISTIC REGRESSION"

log_reg_model = LogisticRegression(C=0.01, class_weight='balanced', multi_class='ovr', random_state=42)
log_reg_model.fit(coord_train, step_train)

'''# Predictions on training data
train_predictions = log_reg_model.predict(coord_train)
print("\nLogReg Classification Report for Training Set \n", classification_report(step_train, train_predictions, zero_division=0))
# Predictions on test data
test_predictions = log_reg_model.predict(coord_test)
print("\nLogReg Classification Report for Test Set \n", classification_report(step_test, test_predictions, zero_division=0))'''

# Define hyperparameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [100, 250, 500, 1000],
    'class_weight': ['balanced', None]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_lr = GridSearchCV(estimator=log_reg_model, param_grid=param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_lr.fit(coord_train, step_train)

# Retrieve the best hyperparameters from GridSearchCV
best_params_lr = grid_search_lr.best_params_
print("\nBest Hyperparameters for Logistic Regression Model:\n", best_params_lr)

# Get the best model from GridSearchCV
best_log_reg_model = grid_search_lr.best_estimator_

# Predict on the test data using the best model
log_reg_final_pred = best_log_reg_model.predict(coord_test)

# Performance Analysis
log_reg_accuracy_score = accuracy_score(step_test, log_reg_final_pred)
log_reg_f1_score = f1_score(step_test, log_reg_final_pred, average ='weighted')
log_reg_classification_report = classification_report(step_test, log_reg_final_pred)

print("\nModel Performance Analysis - Logistic Regression:")
print("\nAccuracy Score:", log_reg_accuracy_score)
print("\nf1 Accuracy Score:", log_reg_f1_score)
print("\nLogisitc Regression Classification Report:\n", log_reg_classification_report)

# Additional Accuracy Check for Training Set (LogReg)
log_reg_train_pred = best_log_reg_model.predict(coord_train)
train_accuracy_lr = accuracy_score(step_train, log_reg_train_pred)
print(f"Logistic Regression Training Accuracy: {train_accuracy_lr:.4f}")

# Accuracy Check for Test Set (LogReg)
log_reg_test_pred = best_log_reg_model.predict(coord_test)
test_accuracy_lr = accuracy_score(step_test, log_reg_test_pred)
print(f"Logistic Regression Testing Accuracy: {test_accuracy_lr:.4f}")
print("\n---------------------------------------------------------------------------------------\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"Model 2 - RANDOM FOREST"

rf_model = RandomForestClassifier(random_state=42, 
                                       max_depth=5, 
                                       min_samples_split=45,
                                       min_samples_leaf=40,
                                       n_estimators=10, 
                                       max_features='sqrt',
                                       class_weight='balanced')
rf_model.fit(coord_train, step_train)                                                         

# Predictions on training data
rf_train_predictions = rf_model.predict(coord_train)
#print("\nRandomForest Classification Report for Training Set \n", classification_report(step_train,         
                                                                        # rf_train_predictions, 
                                                                         #zero_division=0))
# Predictions on test data
rf_test_predictions = rf_model.predict(coord_test)                                            
#print("\nRandomForest Classification Report for Test Set \n", classification_report(step_test,  
                                                                  #   rf_test_predictions, 
                                                                 #    zero_division=0))'''
# Define hyperparameter grid for Random Forest
param_grid_rf = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
 }

# Perform GridSearchCV for hyperparameter tuning
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_rf.fit(coord_train, step_train)

# Retrieve the best hyperparameters from GridSearchCV
best_params_rf = grid_search_rf.best_params_
print("\nBest Hyperparameters for Random Forest Model:\n", best_params_rf)

# Get the best model from grid search
best_rf_model = grid_search_rf.best_estimator_

# Predict on the test data using the best model
final_predictions = best_rf_model.predict(coord_test)

# Performance Analysis
accuracy_score_rf = accuracy_score(step_test, final_predictions)
rf_f1_score = f1_score(step_test, final_predictions, average ='weighted')
classification_report_rf = classification_report(step_test, final_predictions)

# Print accuracy score and classification report
print("\nModel Performance Analysis - Random Forest\n")
print("\nAccuracy Score:", accuracy_score_rf)
print("\nf1 Accuracy Score:", rf_f1_score)
print("\nRandom Forest Classification Report:\n", classification_report_rf)

# Additional Accuracy Check for Training Set (Random Forest)
rf_train_pred = best_rf_model.predict(coord_train)
train_accuracy_rf = accuracy_score(step_train, rf_train_pred)
print(f"Random Forest Training Accuracy: {train_accuracy_rf:.4f}")

# Accuracy Check for Test Set (Random Forest)
rf_test_pred = best_rf_model.predict(coord_test)
test_accuracy_rf = accuracy_score(step_test, rf_test_pred)
print(f"Random Forest Testing Accuracy: {test_accuracy_rf:.4f}")
print("\n---------------------------------------------------------------------------------------\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"Model 3 - SVM (Support Vector Machine)"

svm_model = SVC(random_state=42,                                               # Initialize SVM model with class weight balanced
                class_weight='balanced')

svm_model.fit(coord_train, step_train)                                         # Train the model using the training data

svm_pred_train = svm_model.predict(coord_train)                                # Predict on the training data and evaluate performance
#print("\nSVM Classification Report for Training Set \n", classification_report(step_train, 
                                                                #   svm_pred_train, 
                                                                #  zero_division=0))

svm_pred_test = svm_model.predict(coord_test)                                  # Predict on the test data and evaluate performance
#print("\nSWM Classification Report for Test Set \n", classification_report(step_test, 
                                                                # svm_pred_test, 
                                                                # zero_division=0))
# Define hyperparameter grid for SVM
param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf'],
    'class_weight': ['balanced', None]
}

# Perform GridSearchCV for Hyperparameter Tuning
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_svm.fit(coord_train, step_train)

# Retrieve the Best Hyperparameters from GridSearchCV
best_params_svm = grid_search_svm.best_params_
print("\nBest Hyperparameters for SVM Model:\n", best_params_svm)

# Get the Best Model from GridSearchCV
best_svm_model = grid_search_svm.best_estimator_

# Predict on the Test Data using the Best Model and Evaluate Performance
svm_final_pred = best_svm_model.predict(coord_test)

# Performance Analysis
svm_accuracy_score = accuracy_score(step_test, svm_final_pred)
svm_f1_score = f1_score(step_test, svm_final_pred, average ='weighted')
svm_confusion_matrix = confusion_matrix(step_test, svm_final_pred)
svm_classification_report = classification_report(step_test, svm_final_pred, zero_division=0)

print("\nModel Performance Analysis - Support Vector Machine\n")
print("\nAccuracy Score:", svm_accuracy_score)
print("\nf1 Accuracy Score:", svm_f1_score)
print("\nClassification Report:\n", svm_classification_report)

# Additional Accuracy Check for Training Set (Support Vector Machine)
svm_train_pred = best_svm_model.predict(coord_train)
train_accuracy_svm = accuracy_score(step_train, svm_train_pred)
print(f"Support Vector Machine Training Accuracy: {train_accuracy_svm:.4f}")

# Accuracy Check for Test Set (Support Vector Machine)
svm_test_pred = best_svm_model.predict(coord_test)
test_accuracy_svm = accuracy_score(step_test, svm_test_pred)
print(f"Support Vector Machine Testing Accuracy: {test_accuracy_svm:.4f}")
print("\n---------------------------------------------------------------------------------------\n")


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"Model 4 - Decision Tree"

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(coord_train, step_train)
decision_tree_predictions = decision_tree_model.predict(coord_test)

param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_dt = RandomizedSearchCV(decision_tree_model, param_grid_dt, cv=5, scoring='f1_weighted', n_jobs=1)
random_search_dt.fit(coord_train, step_train)
best_params_dt = random_search_dt.best_params_
best_dt_model = random_search_dt.best_estimator_

# Use the best model for predictions
decision_tree_predictions = best_dt_model.predict(coord_test)

# Performance Analysis
decision_tree_accuracy = accuracy_score(step_test, decision_tree_predictions)
decision_tree_classification_report = classification_report(step_test, decision_tree_predictions)
dt_f1_score = f1_score(step_test, decision_tree_predictions, average ='weighted')

print("\nModel 4 Performance Analysis: Decision Tree - Randomized Search")
print("\nBest Hyperparameters:", best_params_dt)
print("\nAccuracy Score:", decision_tree_accuracy)
print("\nf1 Accuracy Score:", dt_f1_score)
print("\nClassification Report:\n", decision_tree_classification_report)

# Additional Accuracy Check for Training Set (Decision Tree)
dt_train_pred = best_dt_model.predict(coord_train)
train_accuracy_dt = accuracy_score(step_train, dt_train_pred)
print(f"Decision Tree Training Accuracy: {train_accuracy_dt:.4f}")

# Accuracy Check for Test Set (Decision Tree)
dt_test_pred = best_dt_model.predict(coord_test)
test_accuracy_dt = accuracy_score(step_test, dt_test_pred)
print(f"Decision Tree Testing Accuracy: {test_accuracy_dt:.4f}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.5) STEP 5: Model Performance Analysis"

# Plot the Confusion Matrix
svm_confusion_matrix = confusion_matrix(step_test, svm_final_pred)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM")
plt.show()

# print("\nConfusion Matrix:\n", svm_confusion_matrix)
print("\nSummary:")
print("\nAccuracy Score for Logistic Regression:", log_reg_accuracy_score)
print("\nAccuracy Score for Random Forest:", accuracy_score_rf)
print("\nAccuracy Score SVM:", svm_accuracy_score)
print("\n---------------------------------------------------------------------------------------\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.6) STEP  6: Stacked Model Performance Analysis"

# Combining the Models
combined_model = [('SVM', best_svm_model), ('RandomForest', best_rf_model)]

# Defining the Final Model
final_model = LogisticRegression(max_iter=500)

# Creating the Stacking Classifier
stacked_model = StackingClassifier(estimators=combined_model, final_estimator=final_model, cv=5)
stacked_model.fit(coord_train, step_train) 

# Predicting with the Stacked Model
stacked_model_pred = stacked_model.predict(coord_test)  

# Evaluating the Performance
stacked_model_accuracy_score = accuracy_score(step_test, stacked_model_pred) 
stacked_f1_score = f1_score(step_test, stacked_model_pred, average ='weighted')
precision = precision_score(step_test, stacked_model_pred, average='weighted')
accuracy = accuracy_score(step_test, stacked_model_pred)

stacked_model_confusion_matrix = confusion_matrix(step_test, stacked_model_pred)
stacked_model_classification_report = classification_report(step_test, stacked_model_pred)

# Plot the Confusion Matrix
stacked_confusion_matrix = confusion_matrix(step_test, stacked_model_pred)
disp_stacked = ConfusionMatrixDisplay(confusion_matrix=stacked_confusion_matrix)
disp_stacked.plot(cmap=plt.cm.Blues)
plt.title("StackingClassifier Confusion Matrix")
plt.show()

# Printing the Performance Analysis
print("\nStacked Model Performance Analysis")
print("\nAccuracy Score:", stacked_model_accuracy_score)
print("\nf1 Accuracy Score:", stacked_f1_score)

print("\nConfusion Matrix:\n", stacked_model_confusion_matrix)
print("\nClassification Report:\n", stacked_model_classification_report)

print("\nStacked f1 Score:", stacked_f1_score)
print("\nStacked Precision Score:", stacked_f1_score)
print("\nStacked Accuracy Score:", stacked_f1_score)
print("\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.7) STEP  7: Model Evaluation"

# Save the best Random Forest Model
joblib.dump(best_rf_model, 'best_rf_model.joblib')

# Load the saved Random Forest Model
loaded_rf_model = joblib.load('best_rf_model.joblib')

# Coordinates for prediction
coordinates_to_predict = pd.DataFrame([[9.375, 3.0625, 1.51],
                                       [6.995, 5.125, 0.3875],
                                       [0, 3.0625, 1.93],
                                       [9.4, 3, 1.8],
                                       [9.4, 3, 1.3]],
                                      columns=['X', 'Y', 'Z'])

# Scale the prediction data
scaled_data = coord_scaler.transform(coordinates_to_predict)

# Ensure the scaled data has the same feature names as the training data
dataframe_scaled = pd.DataFrame(scaled_data, columns=coord_train.columns)

# Predict using the loaded model
predictions = loaded_rf_model.predict(dataframe_scaled)

print("\n---------------------------------------------------------------------------------------\n")
print("Predicted Step for Sample Coordinates:\n")
print(predictions)
    













 
