#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import json
import sys
import os

def load_historical_data():
    """Load historical data from the core analysis"""
    try:
        df = pd.read_csv('../betler_cost_analysis/output/cost_analysis.csv')
        df['date'] = pd.to_datetime(df['date'])

        # Remove outlier dates (same as in cost_prediction.py)
        OUTLIER_DATES = [
            '2025-05-01',    # Negative cost (-$3,737) - AWS billing correction
            '2025-04-17',    # Customer volume anomaly (5M instead of 10M+)
        ]
        df = df[~df['date'].dt.strftime('%Y-%m-%d').isin(OUTLIER_DATES)]
        df_clean = df.dropna()

        print(f"Loaded {len(df_clean)} historical data points")
        return df_clean

    except FileNotFoundError:
        print("Error: Historical cost_analysis.csv not found. Please run core analysis first.")
        sys.exit(1)

def load_regression_model():
    """Load the core regression model parameters"""
    try:
        with open('../betler_cost_analysis/output/core_regression_model.json', 'r') as f:
            model = json.load(f)
        return model
    except FileNotFoundError:
        print("Error: Core regression model not found. Please run core analysis first.")
        sys.exit(1)

def generate_predictive_data(historical_df, model, customer_growth_rate=0.03, transaction_growth_rate=0.05, months=12):
    """Generate predictive data for future months"""

    # Get the latest values as baseline
    latest_date = historical_df['date'].max()
    latest_customers = historical_df[historical_df['date'] == latest_date]['customer_volume'].iloc[0]
    latest_transactions = historical_df[historical_df['date'] == latest_date]['transaction_volume'].iloc[0]

    # Generate daily predictions for future months
    future_data = []
    current_date = latest_date + timedelta(days=1)

    # Calculate daily growth rates from monthly rates
    daily_customer_growth = (1 + customer_growth_rate) ** (1/30) - 1
    daily_transaction_growth = (1 + transaction_growth_rate) ** (1/30) - 1

    for day in range(months * 30):  # Approximate months as 30 days each
        # Apply compound daily growth
        projected_customers = latest_customers * ((1 + daily_customer_growth) ** day)
        projected_transactions = latest_transactions * ((1 + daily_transaction_growth) ** day)

        # Predict cost using regression model
        predicted_cost = (model['model']['intercept'] +
                         model['model']['transaction_coefficient'] * projected_transactions +
                         model['model']['customer_coefficient'] * projected_customers)

        future_data.append({
            'date': current_date + timedelta(days=day),
            'transaction_volume': projected_transactions,
            'customer_volume': projected_customers,
            'aws_cost': predicted_cost
        })

    future_df = pd.DataFrame(future_data)
    print(f"Generated {len(future_df)} predictive data points")
    return future_df

def create_extended_dashboard(historical_df, future_df):
    """Create comprehensive 24-month dashboard - single plot with multiple axes"""

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.suptitle('Betler Platform: Extended 24-Month Dashboard\n(12 Months Historical + 12 Months Predictive)',
                 fontsize=16, fontweight='bold')

    # Today's date for reference line
    today = datetime.now()

    # Primary axis: Transaction Volume (left y-axis)
    color = 'tab:blue'
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Transaction Volume', color=color)

    # Historical transaction data
    ax.plot(historical_df['date'], historical_df['transaction_volume'],
            color='blue', linewidth=2, alpha=0.8, label='Transactions (Historical)')

    # Future transaction predictions
    ax.plot(future_df['date'], future_df['transaction_volume'],
            color='blue', linewidth=2, alpha=0.6, linestyle='--', label='Transactions (Predicted)')

    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3)

    # Second axis: Customer Volume (right y-axis)
    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Daily Customer Volume', color=color)

    # Historical customer data
    ax2.plot(historical_df['date'], historical_df['customer_volume'],
             color='green', linewidth=2, alpha=0.8, label='Customers (Historical)')

    # Future customer predictions
    ax2.plot(future_df['date'], future_df['customer_volume'],
             color='green', linewidth=2, alpha=0.6, linestyle='--', label='Customers (Predicted)')

    ax2.tick_params(axis='y', labelcolor=color)

    # Third axis: AWS Cost (right y-axis, offset)
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    color = 'tab:red'
    ax3.set_ylabel('Daily AWS Cost (USD)', color=color)

    # Historical cost data
    ax3.plot(historical_df['date'], historical_df['aws_cost'],
             color='red', linewidth=2, alpha=0.8, label='Cost (Historical)')

    # Future cost predictions
    ax3.plot(future_df['date'], future_df['aws_cost'],
             color='red', linewidth=2, alpha=0.6, linestyle='--', label='Cost (Predicted)')

    ax3.tick_params(axis='y', labelcolor=color)

    # Add vertical line for "today" boundary
    ax.axvline(x=today, color='gray', linestyle='-', alpha=0.7, linewidth=3, label='Today')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Combined legend from all axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(f'{output_dir}/extended_24month_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Extended dashboard saved as: {output_dir}/extended_24month_dashboard.png")

    return fig

def generate_summary_stats(historical_df, future_df):
    """Generate summary statistics for the extended analysis"""

    print("\n" + "="*70)
    print("EXTENDED 24-MONTH ANALYSIS SUMMARY")
    print("="*70)

    # Date ranges
    hist_start = historical_df['date'].min().strftime('%Y-%m-%d')
    hist_end = historical_df['date'].max().strftime('%Y-%m-%d')
    future_start = future_df['date'].min().strftime('%Y-%m-%d')
    future_end = future_df['date'].max().strftime('%Y-%m-%d')

    print(f"Historical Period: {hist_start} to {hist_end} ({len(historical_df)} days)")
    print(f"Predictive Period: {future_start} to {future_end} ({len(future_df)} days)")

    # Current vs Future Comparison
    print(f"\nCURRENT VS FUTURE COMPARISON:")
    current_transactions = historical_df['transaction_volume'].iloc[-1]
    current_customers = historical_df['customer_volume'].iloc[-1]
    current_cost = historical_df['aws_cost'].iloc[-1]

    future_transactions = future_df['transaction_volume'].iloc[-1]
    future_customers = future_df['customer_volume'].iloc[-1]
    future_cost = future_df['aws_cost'].iloc[-1]

    print(f"Current (Latest Historical):")
    print(f"  Transactions: {current_transactions:,.0f}/day")
    print(f"  Customers: {current_customers:,.0f}")
    print(f"  AWS Cost: ${current_cost:,.2f}/day")

    print(f"\nProjected (12 Months Future):")
    print(f"  Transactions: {future_transactions:,.0f}/day ({((future_transactions/current_transactions)-1)*100:.1f}% growth)")
    print(f"  Customers: {future_customers:,.0f} ({((future_customers/current_customers)-1)*100:.1f}% growth)")
    print(f"  AWS Cost: ${future_cost:,.2f}/day ({((future_cost/current_cost)-1)*100:.1f}% growth)")

    # Cost projections
    historical_monthly_avg = historical_df['aws_cost'].mean() * 30
    future_monthly_avg = future_df['aws_cost'].mean() * 30

    print(f"\nMONTHLY COST PROJECTIONS:")
    print(f"Historical Average: ${historical_monthly_avg:,.2f}/month")
    print(f"Future Average: ${future_monthly_avg:,.2f}/month")
    print(f"Future Annual Projection: ${future_monthly_avg * 12:,.2f}/year")

def main():
    """Main execution function"""
    print("Loading historical data...")
    historical_df = load_historical_data()

    print("Loading regression model...")
    model = load_regression_model()

    print("Generating predictive data...")
    # Using default 3% customer growth and 5% transaction growth per month
    future_df = generate_predictive_data(historical_df, model,
                                       customer_growth_rate=0.03,
                                       transaction_growth_rate=0.05,
                                       months=12)

    print("Creating extended dashboard...")
    fig = create_extended_dashboard(historical_df, future_df)

    print("Generating summary statistics...")
    generate_summary_stats(historical_df, future_df)

    print("\n" + "="*70)
    print("EXTENDED DASHBOARD COMPLETE")
    print("="*70)
    print("Files created:")
    print("  - output/extended_24month_dashboard.png")
    print("\nDashboard shows 12 months historical + 12 months predictive data")
    print("Red vertical line indicates current date boundary")

if __name__ == "__main__":
    main()