# Heart Disease Prediction using Decision Trees and Random Forest

This project is part of an AI & ML internship task focused on building and evaluating tree-based classification models using the Heart Disease dataset.

## Objective

To build machine learning models that predict the presence of heart disease using:
- Decision Tree
- Pruned Decision Tree
- Random Forest

And to compare their performances using accuracy and feature importance analysis.

## Dataset

The dataset used is the Heart Disease dataset from Kaggle or UCI Machine Learning Repository. It includes features like age, sex, blood pressure, cholesterol levels, etc., with a target variable indicating presence (1) or absence (0) of heart disease.

## What This Notebook Does

1. Loads and preprocesses the dataset
2. Handles duplicates and scales features
3. Trains a:
   - Full Decision Tree
   - Pruned Decision Tree (max_depth=3)
   - Random Forest Classifier
4. Visualizes the decision trees using Graphviz
5. Evaluates models using:
   - Accuracy
   - Classification Report (Precision, Recall, F1-Score)
   - Cross-validation (Random Forest)
6. Plots feature importance for Random Forest
7. Compares model accuracy using bar plots

## Results

| Model                 | Accuracy (Test Set)   |
|-----------------------|-----------------------|
| Decision Tree         | 0.7377049180327869    |
| Pruned Decision Tree  | 0.7540983606557377    |
| Random Forest         | 0.8360655737704918    |

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- graphviz

## Visualization

- Tree structures saved as PDF files using Graphviz:
  - `decision_tree_heart.pdf`
  - `pruned_decision_tree.pdf`
- Feature importance plotted for the Random Forest
- Model comparison using bar charts

## How to Run

1. Make sure you have Python installed.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn graphviz
