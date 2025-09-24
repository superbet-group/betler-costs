#!/usr/bin/env python3

import csv
import numpy as np
from datetime import datetime
import sys

def load_monthly_csv_data(filename):
    """Load monthly data from CSV file and return clean numerical data"""
    # Optimization cutoff: July 2025 authentication optimization reduced costs by 60%
    OPTIMIZATION_CUTOFF = '2025-07'
    POST_OPTIMIZATION_FACTOR = 0.4  # Post-optimization costs are 40% of pre-optimization

    months = []
    transaction_volumes = []
    customer_volumes = []
    cognito_costs = []
    pre_optimization_months = []
    pre_optimization_transactions = []
    pre_optimization_customers = []
    pre_optimization_costs = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip rows with missing data
            if not all([row['monthly_transaction_volume'], row['monthly_customer_volume'], row['cognito_cost']]):
                continue

            try:
                month = row['month']
                transaction_vol = float(row['monthly_transaction_volume'])
                customer_vol = float(row['monthly_customer_volume'])
                cost = float(row['cognito_cost'])

                months.append(month)
                transaction_volumes.append(transaction_vol)
                customer_volumes.append(customer_vol)
                cognito_costs.append(cost)

                # Separate pre-optimization data for model training
                if month < OPTIMIZATION_CUTOFF:
                    pre_optimization_months.append(month)
                    pre_optimization_transactions.append(transaction_vol)
                    pre_optimization_customers.append(customer_vol)
                    pre_optimization_costs.append(cost)

            except ValueError:
                # Skip rows with invalid numerical data
                continue

    return (months, transaction_volumes, customer_volumes, cognito_costs,
            pre_optimization_months, pre_optimization_transactions,
            pre_optimization_customers, pre_optimization_costs, POST_OPTIMIZATION_FACTOR)

def multiple_linear_regression(X, y):
    """
    Perform multiple linear regression using normal equation
    X: feature matrix (n_samples x n_features)
    y: target values (n_samples,)
    Returns: coefficients, intercept, r_squared
    """
    # Add intercept term (column of ones)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # Normal equation: theta = (X^T * X)^(-1) * X^T * y
    theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)

    # Extract intercept and coefficients
    intercept = theta[0]
    coefficients = theta[1:]

    # Calculate R-squared
    y_pred = X_with_intercept @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return coefficients, intercept, r_squared, y_pred

def main():
    """Main execution function"""
    print("Loading monthly CSV data...")

    # Load the merged monthly data
    (months, transaction_volumes, customer_volumes, cognito_costs,
     pre_opt_months, pre_opt_transactions, pre_opt_customers, pre_opt_costs,
     optimization_factor) = load_monthly_csv_data('output/cognito_cost_analysis.csv')

    print(f"Loaded {len(months)} total monthly data points")
    print(f"Pre-optimization training data: {len(pre_opt_months)} months (before 2025-07)")
    print(f"Optimization factor: {optimization_factor:.2f} (post-optimization costs)")

    if len(pre_opt_months) < 3:
        print("Error: Need at least 3 months of pre-optimization data for meaningful regression analysis")
        sys.exit(1)

    # Train model on pre-optimization data (more stable relationship)
    print("\nTraining model on pre-optimization data for stable cost relationships...")
    X_train = np.column_stack([pre_opt_transactions, pre_opt_customers])
    y_train = np.array(pre_opt_costs)

    # Perform regression on pre-optimization data
    coefficients, intercept, r_squared_train, train_predictions = multiple_linear_regression(X_train, y_train)

    # Generate hybrid predictions: pre-optimization unchanged, post-optimization * 0.4
    X_all = np.column_stack([transaction_volumes, customer_volumes])
    base_predictions = intercept + X_all @ coefficients

    hybrid_predictions = []
    for i, month in enumerate(months):
        if month < '2025-07':
            # Pre-optimization: use base prediction
            hybrid_predictions.append(base_predictions[i])
        else:
            # Post-optimization: apply 60% reduction (multiply by 0.4)
            hybrid_predictions.append(base_predictions[i] * optimization_factor)

    hybrid_predictions = np.array(hybrid_predictions)

    # Display results
    print("\n" + "="*60)
    print("COGNITO COST REGRESSION RESULTS (HYBRID MODEL)")
    print("="*60)
    print("Model Strategy:")
    print("- Trained on pre-optimization data for stable cost relationships")
    print("- Pre-July 2025: Use base model predictions")
    print(f"- July 2025+: Apply {optimization_factor:.1f} multiplier (60% cost reduction)")
    print("- Uses predictable business metrics (customers, transactions)")
    print()
    print(f"Base Model: Cognito_Cost = {intercept:.2f} + {coefficients[0]:.8f} * Transactions + {coefficients[1]:.4f} * Customers")
    print(f"Post-July 2025: Predicted_Cost = Base_Cost * {optimization_factor:.1f}")
    print(f"Training R-squared: {r_squared_train:.4f} ({r_squared_train*100:.2f}% of pre-optimization variance explained)")

    # Calculate statistics for hybrid predictions vs actual data
    y_actual = np.array(cognito_costs)
    residuals = y_actual - hybrid_predictions
    rmse = np.sqrt(np.mean(residuals**2))
    mean_cost = np.mean(y_actual)

    print(f"\nHybrid Model Performance (All Data):")
    print(f"Root Mean Square Error: ${rmse:,.2f}")
    print(f"Mean Actual Cognito Cost: ${mean_cost:,.2f}")
    print(f"RMSE as % of mean: {(rmse/mean_cost)*100:.2f}%")

    # Show predictions vs actual with optimization indicator
    print(f"\nMonthly Predictions vs Actual (Hybrid Model):")
    print("Month\t\tActual\t\tPredicted\tDifference\tPeriod")
    print("-" * 80)
    for i in range(len(months)):
        actual = y_actual[i]
        predicted = hybrid_predictions[i]
        diff = actual - predicted
        period = "Pre-opt" if months[i] < '2025-07' else "Post-opt"
        print(f"{months[i]}\t${actual:,.2f}\t${predicted:,.2f}\t${diff:,.2f}\t{period}")

    # Example predictions (using optimized model)
    print(f"\n" + "="*60)
    print("EXAMPLE COGNITO COST PREDICTIONS (POST-OPTIMIZATION)")
    print("="*60)

    # Use recent values as baseline
    recent_transactions = transaction_volumes[-1]
    recent_customers = customer_volumes[-1]

    scenarios = [
        ("Current levels", recent_transactions, recent_customers),
        ("10% more transactions", recent_transactions * 1.1, recent_customers),
        ("10% more customers", recent_transactions, recent_customers * 1.1),
        ("10% growth in both", recent_transactions * 1.1, recent_customers * 1.1),
        ("20% growth in both", recent_transactions * 1.2, recent_customers * 1.2),
    ]

    print("Future predictions use post-optimization factor (60% cost reduction):")
    for scenario_name, trans_vol, cust_vol in scenarios:
        base_cost = intercept + coefficients[0] * trans_vol + coefficients[1] * cust_vol
        post_opt_cost = base_cost * optimization_factor
        print(f"{scenario_name:20}: ${post_opt_cost:,.2f} (base: ${base_cost:,.2f})")


    # Save regression model as JSON for predictive analysis
    model_json = {
        "model_type": "hybrid_cognito",
        "methodology": "Pre-optimization regression with post-July 2025 optimization factor",
        "training_period": f"{pre_opt_months[0]} to {pre_opt_months[-1]}",
        "training_months": len(pre_opt_months),
        "total_months": len(months),
        "optimization_cutoff": "2025-07",
        "optimization_factor": optimization_factor,
        "base_model": {
            "intercept": float(intercept),
            "transaction_coefficient": float(coefficients[0]),
            "customer_coefficient": float(coefficients[1]),
            "r_squared": float(r_squared_train),
            "equation": f"Cognito_Cost = {intercept:.4f} + {coefficients[0]:.8f} * Monthly_Transactions + {coefficients[1]:.4f} * Monthly_Customers"
        },
        "hybrid_model": {
            "rmse": float(rmse),
            "rmse_percent": float((rmse/mean_cost)*100),
            "mean_actual_cost": float(mean_cost)
        },
        "cost_drivers": {
            "per_transaction_pre_opt": float(coefficients[0]),
            "per_customer_pre_opt": float(coefficients[1]),
            "per_transaction_post_opt": float(coefficients[0] * optimization_factor),
            "per_customer_post_opt": float(coefficients[1] * optimization_factor)
        },
        "generated_date": datetime.now().isoformat()
    }

    import json
    with open('output/cognito_regression_model.json', 'w') as f:
        json.dump(model_json, f, indent=2)

    print(f"Regression model saved to: output/cognito_regression_model.json")

if __name__ == "__main__":
    main()