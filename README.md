# Body Fat Prediction Project

## Project Overview

This project focuses on developing and evaluating machine learning models to predict body fat percentage based on various health and physical attributes. The goal is to leverage different algorithms to achieve accurate predictions and gain insights into feature importance and model performance.

## Objectives

- **Develop Predictive Models:** Create and fine-tune machine learning models to predict body fat percentage.
- **Feature Engineering:** Engineer new features to improve model performance and interpretability.
- **Evaluate Models:** Assess models using various metrics to ensure robustness and reliability.

## Dataset

The dataset comprises various physical attributes such as weight, height, age, and measurements from different body parts. The dataset is preprocessed and split into training and testing sets for model evaluation.

## Models Used

1. **Feedforward Neural Network (FNN)**
   - **Architecture:** 2 hidden layers (16, 8 units)
   - **Activation Function:** ReLU
   - **Optimizer:** Adam
   - **Loss Function:** Mean Squared Error
   - **Early Stopping:** Patience: 5, Min Delta: 0.001, Restore Best Weights

2. **Deep Neural Network with Dropout**
   - **Architecture:** 3 hidden layers with dropout
   - **Dropout Rate:** 0.5
   - **Activation Function:** ReLU
   - **Optimizer:** Adam
   - **Loss Function:** Mean Squared Error

3. **XGBoost (XGB)**
   - **Boosting Type:** Gradient Boosting
   - **Key Hyperparameters:** Learning rate, max depth, n_estimators
   - **Performance:** Outperformed other models with superior accuracy and lower error metrics

4. **Random Forest**
   - **Ensemble Method:** Bagging
   - **Key Hyperparameters:** Number of trees, max features
   - **Performance:** Strong performance, though not as optimal as XGBoost

5. **Gradient Boosting**
   - **Boosting Type:** Gradient Boosting
   - **Key Hyperparameters:** Learning rate, number of boosting stages
   - **Performance:** Competitive performance, but slightly less accurate compared to XGBoost

## Feature Engineering

- **Body Mass Index (BMI):** `BMI = Weight / (Height / 100) ** 2`
- **Waist-to-Hip Ratio:** `WaistToHipRatio = Abdomen / Hip`
- **Body Surface Area (BSA):** `BodySurfaceArea = 0.007184 * (Height ** 0.725) * (Weight ** 0.425)`
- **Age Squared:** `AgeSquared = Age ** 2`
- **Abdomen-to-Chest Ratio:** `AbdomenToChestRatio = Abdomen / Chest`
- **Upper Body Fat:** Sum of neck, chest, and biceps measurements
- **Lower Body Fat:** Sum of thigh, knee, and ankle measurements
- **Arm Fat Index:** `(Biceps + Forearm) / Wrist`

## Metrics for Model Evaluation

- **R² Score:** Measures the proportion of variance explained by the model.
- **MAE (Mean Absolute Error):** Average magnitude of prediction errors.
- **MSE (Mean Squared Error):** Average of squared prediction errors.
- **RMSE (Root Mean Squared Error):** Square root of MSE, showing error magnitude in target units.

## Results

### Model Performance Summary

- **XGBoost Model:** Best-performing model with the highest accuracy and lowest error metrics.
- **Other Models:** Feedforward Neural Network, Deep Neural Network with Dropout, Random Forest, and Gradient Boosting showed strong performance but with varying degrees of accuracy.

## Code and Files

- `data_preprocessing.py`: Script for data cleaning and feature engineering.
- `model_training.py`: Script for training machine learning models.
- `model_evaluation.py`: Script for evaluating model performance.
- `README.md`: This file.

## Usage

1. Clone the repository: `git clone https://github.com/yourusername/body-fat-prediction.git`
2. Install required packages: `pip install -r requirements.txt`
3. Run the scripts to preprocess data, train models, and evaluate performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for providing tools and libraries used in this project.
- Special thanks to [Your Institution/Company] for providing support and resources.

