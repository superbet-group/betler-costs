#!/usr/bin/env python3

import csv
import numpy as np
from datetime import datetime
import sys

def load_csv_data(filename):
    """Load data from CSV file and return clean numerical data"""
    # Hardcoded list of outlier dates to exclude from analysis
    OUTLIER_DATES = [
        '2025-05-01',    # Negative cost (-$3,737) - AWS billing correction
        '2025-04-17',    # Customer volume anomaly (5M instead of 10M+)
    ]

    dates = []
    transaction_volumes = []
    customer_volumes = []
    aws_costs = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip outlier dates
            if row['date'] in OUTLIER_DATES:
                continue

            # Skip rows with missing data
            if not all([row['transaction_volume'], row['customer_volume'], row['aws_cost']]):
                continue

            try:
                dates.append(row['date'])
                transaction_volumes.append(float(row['transaction_volume']))
                customer_volumes.append(float(row['customer_volume']))
                aws_costs.append(float(row['aws_cost']))
            except ValueError:
                # Skip rows with invalid numerical data
                continue

    return dates, transaction_volumes, customer_volumes, aws_costs

def multiple_linear_regression(X, y):
    """
    Perform multiple linear regression using normal equation
    X: feature matrix (n_samples x n_features)
    y: target values (n_samples,)
    Returns: coefficients, intercept, r_squared
    """
    # Add intercept term (column of ones)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # Normal equation: theta = (X^T X)^(-1) X^T y
    try:
        theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
    except np.linalg.LinAlgError:
        # Use pseudoinverse if matrix is singular
        theta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    intercept = theta[0]
    coefficients = theta[1:]

    # Calculate R-squared
    y_pred = X_with_intercept @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return coefficients, intercept, r_squared, y_pred

def predict_cost(transaction_volume, customer_volume, coefficients, intercept):
    """Predict cost given transaction volume and customer volume"""
    return intercept + coefficients[0] * transaction_volume + coefficients[1] * customer_volume

def main():
    try:
        # Load data
        print("Loading CSV data...")
        dates, transaction_volumes, customer_volumes, aws_costs = load_csv_data('output/cost_analysis.csv')

        if len(dates) < 2:
            print("Error: Not enough data points for regression analysis")
            sys.exit(1)

        print(f"Loaded {len(dates)} data points")

        # Prepare data for regression
        X = np.column_stack([transaction_volumes, customer_volumes])
        y = np.array(aws_costs)

        # Perform multiple linear regression
        print("\nPerforming multiple linear regression...")
        coefficients, intercept, r_squared, predictions = multiple_linear_regression(X, y)

        # Calculate some statistics for validation
        residuals = y - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        mean_cost = np.mean(y)

        # Minimal output - just key validation
        print(f"✓ Core model: R² = {r_squared:.3f}, RMSE = ${rmse:,.0f} ({(rmse/mean_cost)*100:.1f}% of mean)")

        # Data validation check
        if r_squared < 0.7:
            print(f"⚠ Warning: Low R² ({r_squared:.3f}) - model may not be reliable")
        if len(dates) < 300:
            print(f"⚠ Warning: Limited data ({len(dates)} points) - predictions may be less accurate")


        # Save regression model as JSON for predictive analysis
        model_json = {
            "model_type": "core_aws_cost",
            "methodology": "Multiple linear regression on daily AWS costs (excluding tax and Cognito)",
            "data_points": len(dates),
            "date_range": {
                "start": dates[0],
                "end": dates[-1]
            },
            "model": {
                "intercept": float(intercept),
                "transaction_coefficient": float(coefficients[0]),
                "customer_coefficient": float(coefficients[1]),
                "r_squared": float(r_squared),
                "rmse": float(rmse),
                "equation": f"Cost = {intercept:.4f} + {coefficients[0]:.8f} * Transaction_Volume + {coefficients[1]:.4f} * Customer_Volume"
            },
            "cost_drivers": {
                "per_transaction": float(coefficients[0]),
                "per_customer": float(coefficients[1])
            },
            "performance": {
                "rmse": float(rmse),
                "r_squared": float(r_squared),
                "mean_cost": float(np.mean(aws_costs))
            },
            "generated_date": datetime.now().isoformat()
        }

        import json
        with open('output/core_regression_model.json', 'w') as f:
            json.dump(model_json, f, indent=2)

        # Silent save

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()