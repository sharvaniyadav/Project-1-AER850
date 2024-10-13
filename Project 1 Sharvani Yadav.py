# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:21:44 2024

@author: Sharvani Yadav
"""

# Import necessary libraries
import numpy as np  # NumPy is used for numerical operations, mainly arrays and mathematical functions.
import seaborn as sns  # Seaborn is used for creating advanced visualizations, especially statistical graphics.
import pandas as pd  # Pandas is a library used for data manipulation and analysis, particularly for working with dataframes.
import matplotlib.pyplot as plt  # Matplotlib is a plotting library used for creating static, animated, and interactive visualizations.

# Import specific modules from scikit-learn for machine learning tasks
from sklearn.model_selection import train_test_split  # This function is used to split your data into training and testing sets.
from sklearn.preprocessing import StandardScaler  # StandardScaler is used for feature scaling, normalizing data to improve model performance.
from sklearn.linear_model import LogisticRegression  # LogisticRegression is a classification algorithm used for binary/multiclass classification.
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier is an ensemble learning method based on decision trees.
from sklearn.svm import SVC  # Support Vector Classifier (SVC) is a classification algorithm based on finding a decision boundary between classes.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # These metrics help evaluate the performance of classification models.
import joblib  # Joblib is used to save and load models, making it easier to persist and reuse trained models.

# Load the dataset from a CSV file
df = pd.read_csv('Project_1_Data.csv')  # Reads the data from a CSV file and stores it in a pandas DataFrame for manipulation.

# Data Visualization
from mpl_toolkits.mplot3d import Axes3D  # Axes3D is used to create 3D plots for visualizing data in 3 dimensions (X, Y, Z).

# Create a 3D scatter plot to visualize the coordinates (X, Y, Z)
fig = plt.figure()  # Create a new figure for plotting.
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure.

# Plot the data points on the 3D plot using X, Y, and Z coordinates from the dataframe
ax.scatter(df['X'], df['Y'], df['Z'])  # Scatter plot with X, Y, Z columns from the dataframe.
ax.view_init(30, 185)  # Set the view angle for better visibility (30 degrees elevation and 185 degrees azimuth).

# Show the 3D scatter plot
plt.show()  # Display the plot.
