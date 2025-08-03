# Earthquake Analysis & Magnitude Prediction

This repository contains a comprehensive data analysis and machine learning project focused on global earthquake data. The primary goal is to explore patterns in earthquake occurrences and to build a model that can predict their magnitude.

The entire analysis is documented in a Jupyter Notebook (`Earthquake_Analysis_Improved.ipynb`) that serves as an article-style walkthrough of the project.

#### Key Features:

*   **In-Depth Data Analysis:** The project starts with data cleaning and feature engineering, creating new, insightful features from the raw data.
*   **Exploratory Data Analysis (EDA):** Includes a variety of visualizations like histograms, scatter plots, and a correlation heatmap to uncover trends and relationships within the data.
*   **Machine Learning Model:** A `RandomForestRegressor` is implemented using Scikit-learn to predict earthquake magnitudes based on features like latitude, longitude, and depth. The model's performance is evaluated using cross-validation.
*   **Interactive Visualization:** An interactive world map is generated using `folium`, displaying a heatmap of earthquake locations and magnitudes.
*   **Technologies Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Folium.

### How to Run

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `folium`.
3.  Open and run the `Earthquake_Analysis_Improved.ipynb` notebook.