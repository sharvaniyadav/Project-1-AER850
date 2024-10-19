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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score                # These metrics help evaluate the performance of classification models.
import joblib                                                                                      # Joblib is used to save and load models, making it easier to persist and reuse trained models.
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.1) STEP 1: Data Processing"

# Load the dataset from a CSV file
df = pd.read_csv('Project_1_Data.csv') 
df = df.dropna()  # Remove rows with missing values  
                                                      # Reads the data from a CSV file and stores it in a pandas DataFrame for manipulation.
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
plt.show()  # Display the plot.

"This code creates 2D visualization Bar Graph of the data:"
'''This code analyzes and visualizes the distribution of the Step variable in your dataset. 
It counts how many times each unique step appears, creates a bar graph to display these counts,
and adds labels to explain the graph. The final output helps you understand how frequently 
each step occurs, revealing trends or imbalances in your data.'''
 
# Count how many times each step appears
step_count = df["Step"].value_counts()

# Draw a bar graph of the step counts
step_count.plot(kind="bar")

# Add labels to explain the graph
plt.title("Distribution of Steps of Inverter")
plt.xlabel("Step")
plt.ylabel("Number of Instances")

# Show the graph
plt.show()

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

correlation_matrix = coord_train.corr()                                        # Gathers the Correlation Matrix that will be later inputted
sb.heatmap(np.abs(correlation_matrix))                                         # Creates Heatmap to visualize the correlation matrix itself

''' Based on the heatmap, the input variables don’t strongly correlate with 
each other (the highest is around 0.2). This means none of the variables are 
too similar, so none of the data needs to be dropped from the dataset itself.'''

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"(2.4) STEP 4: Classification Model Development/Engineering"

"Training Model 1 - LOGISTIC REGRESSION"

log_reg_model = LogisticRegression(C=0.01, class_weight='balanced', multi_class='ovr', random_state=42) 
log_reg_model.fit(coord_train, step_train)

# Predictions on training data
train_predictions = log_reg_model.predict(coord_train)
print("Classification Report for Training Set \n", classification_report(step_train, 
                                                                         train_predictions, 
                                                                         zero_division=0))
# Predictions on test data
test_predictions = log_reg_model.predict(coord_test)
print("Classification Report for Test Set \n", classification_report(step_test, 
                                                                     test_predictions, 
                                                                     zero_division=0))

"Training Model 2 - RANDOM FOREST"

rf_model = RandomForestClassifier(random_state=42, 
                                       max_depth=5, 
                                       min_samples_split=45,
                                       min_samples_leaf=40,
                                       n_estimators=10, 
                                       max_features='sqrt',
                                       class_weight='balanced')

rf_model.fit(coord_train, step_train)                                                         # Train the model

rf_train_predictions = rf_model.predict(coord_train)                                          # Predictions on training data
print("Classification Report for Training Set \n", classification_report(step_train,          # Print the classification report for the training set
                                                                         rf_train_predictions, 
                                                                         zero_division=0))

rf_test_predictions = rf_model.predict(coord_test)                                            # Predictions on test data
print("Classification Report for Test Set \n", classification_report(step_test,               # Print the classification report for the testing set
                                                                     rf_test_predictions, 
                                                                 zero_division=0))
# Hyperparameter Grid for Tuning
param_grid_rf = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
 }

# Implement Grid Search for Hyperparameter Tuning
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_rf.fit(coord_train, step_train)

# Retrieve the best hyperparameters from the grid search
best_params_rf = grid_search_rf.best_params_
print("Best Hyperparameters for Random Forest Model:", best_params_rf)

# Get the best model from grid search
best_rf_model = grid_search_rf.best_estimator_

# Make predictions with the best model on the test set
final_predictions = best_rf_model.predict(coord_test)


# Performance Analysis
accuracy_score_rf = accuracy_score(step_test, final_predictions)
confusion_matrix_rf = confusion_matrix(step_test, final_predictions)
classification_report_rf = classification_report(step_test, final_predictions)

# Print accuracy score and classification report
print("Model Performance Analysis: Random Forest\n")
print("Accuracy Score:", accuracy_score_rf)
print("\nClassification Report:\n", classification_report_rf)

# Plot the confusion matrix
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forests")
plt.show()

sb.heatmap(confusion_matrix_rf)
plt.show()


















 
