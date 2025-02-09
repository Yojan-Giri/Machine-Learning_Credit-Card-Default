# Bank Marketing Model

This project involves building various machine learning models to predict whether a client will subscribe to a term deposit (yes/no) using the dataset from the Bank Marketing dataset.

## Libraries Used

The following libraries are used in this project:

- `tidyverse`: For data wrangling and visualization.
- `caret`: For model training, testing, and evaluation.
- `e1071`: For SVM models.
- `MASS`: For Linear Discriminant Analysis (LDA).
- `randomForest`: For building random forest models.
- `glmnet`: For fitting generalized linear models.
- `gbm`: For gradient boosting models.
- `smotefamily`: For handling class imbalance using SMOTE.
- `performanceEstimation`: For model performance estimation.
- `class`: For K-Nearest Neighbors (KNN).
- `tree`: For decision tree models.
- `gridExtra`: For arranging multiple plots.

## Dataset Overview

### Bank Marketing Dataset

- **Data Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Description**: The dataset contains information about bank marketing campaigns. The objective is to predict whether a client will subscribe to a term deposit based on various features like age, job, marital status, education, etc.

### Columns:
- **y**: Whether the client subscribed to the term deposit (binary outcome: yes/no).
- **Features**: Includes attributes such as age, job type, marital status, education, loan status, and more.

## Steps

### 1. Data Cleaning
- Filter out unknown values for categorical variables such as job, marital, education, default, loan, housing, and others.
- Convert categorical variables into numeric values where necessary.

### 2. Feature Selection and Transformation
- Created numeric variables for categorical features like job, education, etc., and excluded unnecessary features based on their correlation with the target variable (`y`).

### 3. Exploratory Data Analysis (EDA)
- Visualized data distributions using bar plots.
- Computed correlation matrix for numeric features.

### 4. Model Building
The following models were built and evaluated:

- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Decision Tree (DT)
- Support Vector Machine (SVM) (with multiple kernels)

### 5. Model Evaluation
- Evaluated models using performance metrics such as Accuracy, Sensitivity, Specificity, Precision (PPV), and Negative Predictive Value (NPV).
- Used confusion matrices to measure the prediction performance.

### 6. Hyperparameter Tuning
- Tuning for KNN: Optimized the `k` value for K-Nearest Neighbors.
- Tuning for SVM: Used linear, polynomial, and radial kernels with cross-validation to find the best fit.

### 7. Model Comparison
- Visualized the performance of each model using bar graphs.

## Model Results
### Best Model:
- **K-Nearest Neighbors (KNN)** with `k=81` gave the best performance in terms of accuracy (90.2191%).

## How to Run the Code

1. Install the necessary R packages using the following command:
   ```R
   install.packages(c("tidyverse", "caret", "e1071", "randomForest", "MASS", "smotefamily", "performanceEstimation", "class", "gridExtra", "tree"))
