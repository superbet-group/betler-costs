#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import sys
import os

def load_and_prepare_data():
    """Load CSV data and prepare for plotting"""
    try:
        df = pd.read_csv('output/cost_analysis.csv')

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Define outlier dates to exclude (same as cost_prediction.py)
        OUTLIER_DATES = [
            '2025-05-01',    # Negative cost (-$3,737) - AWS billing correction
            '2025-04-17',    # Customer volume anomaly (5M instead of 10M+)
        ]

        # Remove outlier dates from both dataframes
        df = df[~df['date'].dt.strftime('%Y-%m-%d').isin(OUTLIER_DATES)]

        # Remove rows with missing data for cleaner plots
        df_clean = df.dropna()

        # Data validation
        if len(df_clean) < 300:
            print(f"⚠ Warning: Limited clean data ({len(df_clean)} rows) - visualizations may be incomplete")
        return df, df_clean
    except FileNotFoundError:
        print("Error: cost_analysis.csv not found. Please run the main analysis script first.")
        sys.exit(1)

def create_plots(df, df_clean):
    """Create comprehensive visualization plots"""

    # Create figure with subplots - now 3x2 grid for 6 charts
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Betler Platform: Cost Analysis Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Transaction Volume Over Time
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['transaction_volume'], color='blue', linewidth=1, alpha=0.7)
    ax1.set_title('Daily Transaction Volume', fontweight='bold')
    ax1.set_ylabel('Transactions per Day')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    # Add trend line
    if len(df.dropna(subset=['transaction_volume'])) > 1:
        x_num = mdates.date2num(df.dropna(subset=['transaction_volume'])['date'])
        y_vals = df.dropna(subset=['transaction_volume'])['transaction_volume']
        z = np.polyfit(x_num, y_vals, 1)
        p = np.poly1d(z)
        ax1.plot(df.dropna(subset=['transaction_volume'])['date'], p(x_num),
                "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.0f} transactions/day²')
        ax1.legend()

    # Plot 2: Customer Volume Over Time
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['customer_volume'], color='green', linewidth=1, alpha=0.7)
    ax2.set_title('Customer Volume Growth', fontweight='bold')
    ax2.set_ylabel('Customer Volume')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)

    # Add trend line
    if len(df.dropna(subset=['customer_volume'])) > 1:
        x_num = mdates.date2num(df.dropna(subset=['customer_volume'])['date'])
        y_vals = df.dropna(subset=['customer_volume'])['customer_volume']
        z = np.polyfit(x_num, y_vals, 1)
        p = np.poly1d(z)
        ax2.plot(df.dropna(subset=['customer_volume'])['date'], p(x_num),
                "r--", alpha=0.8, linewidth=2, label=f'Trend: +{z[0]:.0f} customers/day')
        ax2.legend()

    # Plot 3: AWS Daily Costs
    ax3 = axes[1, 0]
    if len(df_clean) > 0:
        ax3.plot(df_clean['date'], df_clean['aws_cost'], color='red', linewidth=1, alpha=0.7)
        ax3.set_title('Daily AWS Costs', fontweight='bold')
        ax3.set_ylabel('Cost (USD)')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.tick_params(axis='x', rotation=45)

        # Add predicted cost line using regression
        try:
            # Calculate predictions using the same regression as in cost_prediction.py
            X = np.column_stack([df_clean['transaction_volume'].values, df_clean['customer_volume'].values])
            y = df_clean['aws_cost'].values

            # Perform regression
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
            predictions = X_with_intercept @ theta

            # Plot predicted costs
            ax3.plot(df_clean['date'], predictions, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                    label='Predicted Cost')
            ax3.legend()
        except Exception:
            # If prediction fails, just show the actual costs without prediction line
            pass
    else:
        ax3.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Daily AWS Costs (No Data)', fontweight='bold')

    # Plot 4: Cost vs Volume Scatter
    ax4 = axes[1, 1]
    if len(df_clean) > 0:
        # Create scatter plot with transaction volume
        scatter = ax4.scatter(df_clean['transaction_volume'], df_clean['aws_cost'],
                            c=df_clean['customer_volume'], cmap='viridis', alpha=0.6, s=50)

        # Add regression line if we have enough data
        if len(df_clean) > 5:
            # Simple linear regression for transaction volume vs cost
            x = df_clean['transaction_volume'].values
            y = df_clean['aws_cost'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax4.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Calculate R-squared
            y_pred = p(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            ax4.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax4.set_title('Cost vs Transaction Volume', fontweight='bold')
        ax4.set_xlabel('Transaction Volume')
        ax4.set_ylabel('AWS Cost (USD)')
        ax4.grid(True, alpha=0.3)

        # Add colorbar for customer volume
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Customer Volume', rotation=270, labelpad=20)
    else:
        ax4.text(0.5, 0.5, 'No cost data available\nfor correlation analysis',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Cost vs Volume Correlation (No Data)', fontweight='bold')

    # Plot 5: Cost per Customer Volume
    ax5 = axes[2, 0]
    if len(df_clean) > 0:
        # Calculate cost per customer (cost / customer_volume * 1000 for per-thousand-customers)
        cost_per_customer = (df_clean['aws_cost'] / df_clean['customer_volume']) * 1000
        ax5.plot(df_clean['date'], cost_per_customer, color='purple', linewidth=1, alpha=0.7)
        ax5.set_title('Cost per 1000 Customers', fontweight='bold')
        ax5.set_ylabel('Cost per 1000 Customers (USD)')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.tick_params(axis='x', rotation=45)

    else:
        ax5.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Cost per Customer (No Data)', fontweight='bold')

    # Plot 6: Actual vs Predicted Costs
    ax6 = axes[2, 1]
    if len(df_clean) > 0:
        # Load regression results to get predictions
        try:
            # Read the regression results to get coefficients
            # For now, let's calculate predictions using the same regression as in cost_prediction.py
            X = np.column_stack([df_clean['transaction_volume'].values, df_clean['customer_volume'].values])
            y = df_clean['aws_cost'].values

            # Perform regression
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
            predictions = X_with_intercept @ theta

            # Plot actual vs predicted
            ax6.scatter(y, predictions, alpha=0.6, color='green', s=50)

            # Add perfect prediction line (diagonal)
            min_val = min(y.min(), predictions.min())
            max_val = max(y.max(), predictions.max())
            ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                    label='Perfect Prediction')

            # Calculate and display R-squared
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            ax6.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax6.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax6.set_title('Actual vs Predicted Costs', fontweight='bold')
            ax6.set_xlabel('Actual Cost (USD)')
            ax6.set_ylabel('Predicted Cost (USD)')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

        except Exception as e:
            ax6.text(0.5, 0.5, f'Error calculating predictions:\n{str(e)}',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=10)
            ax6.set_title('Actual vs Predicted (Error)', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Actual vs Predicted (No Data)', fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(f'{output_dir}/cost_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Core dashboard created")

    return fig

def create_summary_stats(df, df_clean):
    """Generate key validation statistics"""

    # Data validation checks
    if len(df_clean) > 0:
        cost_data = df_clean['aws_cost']
        total_cost = cost_data.sum()

        # Key validation: ensure we have reasonable cost data
        print(f"✓ Core data: ${total_cost:,.0f} total cost over {len(df_clean)} days")

        # Warning checks
        if cost_data.mean() < 1000:
            print("⚠ Warning: Average daily cost unusually low - check data quality")
        if cost_data.max() > 50000:
            print("⚠ Warning: Daily cost spike detected - may affect predictions")
    else:
        print("⚠ Error: No valid cost data found")


def main():
    """Main execution function"""
    df, df_clean = load_and_prepare_data()
    fig = create_plots(df, df_clean)
    create_summary_stats(df, df_clean)

if __name__ == "__main__":
    main()