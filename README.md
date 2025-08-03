# Earthquake Data Analysis and Magnitude Prediction

This project analyzes a dataset of recent earthquakes to understand their characteristics and builds a machine learning model to predict their magnitude.

## Dataset

The dataset used is `all_month.csv`, which contains information about earthquakes that occurred in the last month.

## Analysis

The analysis is performed in the `Earthquake_Analysis.ipynb` Jupyter Notebook. It includes:

- **Exploratory Data Analysis (EDA):** 
    - Loading and cleaning the data.
    - Visualizing the distribution of earthquake magnitudes and depths.
    - Plotting the locations of earthquakes on a world map.
    - Analyzing earthquake trends over time.
- **Machine Learning Model:**
    - A linear regression model is built to predict earthquake magnitude based on latitude, longitude, and depth.
    - The model is evaluated using Mean Squared Error and R-squared metrics.

## Visualizations

The notebook includes the following visualizations:

1.  Distribution of Earthquake Magnitudes
2.  Distribution of Earthquake Depths
3.  Magnitude vs. Depth Scatter Plot
4.  Earthquake Magnitude Over Time
5.  World Map of Earthquakes
6.  Correlation Heatmap
7.  Box plot of magnitude by magType
8.  Violin plot of magnitude by magType
9.  Pairplot of numerical features
10. Earthquakes by hour of the day
11. Actual vs. Predicted Magnitude Scatter Plot

## How to Run

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3.  Open and run the `Earthquake_Analysis.ipynb` notebook.
