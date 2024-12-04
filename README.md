# Machine Learning Experimentation

This project implements and compares various regression models, including a custom random forest regressor, using two datasets from the UCI Machine Learning Repository.

## Overview

The goal of this project is to implement a custom random forest algorithm and compare its performance against existing models such as Scikit-learn's Random Forest, Decision Tree, and K-Nearest Neighbors. The comparison is based on two datasets with different characteristics to evaluate the models' performance across various scenarios.

## Datasets

1. **Dataset 1**: 48.84k samples, 14 features
2. **Dataset 2**: 4.6k samples, 57 features

## Models

- Custom Random Forest
- Scikit-learn Random Forest
- Scikit-learn Decision Tree
- Scikit-learn K-Nearest Neighbors

## Results

The results are visualized in a multi-plot figure saved in the `output` directory. Each subplot represents the performance of a model on a dataset, showing both Mean Squared Error (MSE) and R2 Score.

## Instructions

1. **Install Dependencies**: Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Experiments**: Execute the following command to run the experiments and generate plots:
   ```bash
   python run_experiments.py
   ```

3. **View Results**: Check the `output` directory for the generated plots.

## Conclusion

This project demonstrates the implementation of a custom random forest regressor and its comparison with existing models. The results provide insights into the performance and efficiency of different regression techniques across datasets with varying characteristics.
