#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os
import json
import argparse

def load_existing_analysis_data():
    """Load data from both core and Cognito analysis outputs"""

    # Load core analysis data
    core_data = {}
    try:
        core_csv_path = '../betler_cost_analysis/output/cost_analysis.csv'
        if os.path.exists(core_csv_path):
            core_df = pd.read_csv(core_csv_path)
            core_data['df'] = core_df
            print(f"Loaded core analysis data: {len(core_df)} records")
        else:
            print("Warning: Core analysis data not found. Run core analysis first.")
    except Exception as e:
        print(f"Error loading core analysis data: {e}")

    # Load Cognito analysis data
    cognito_data = {}
    try:
        cognito_csv_path = '../cognito_cost_analysis/output/cognito_cost_analysis.csv'
        if os.path.exists(cognito_csv_path):
            cognito_df = pd.read_csv(cognito_csv_path)
            cognito_data['df'] = cognito_df
            print(f"Loaded Cognito analysis data: {len(cognito_df)} records")
        else:
            print("Warning: Cognito analysis data not found. Run Cognito analysis first.")
    except Exception as e:
        print(f"Error loading Cognito analysis data: {e}")

    return core_data, cognito_data

def load_prediction_models():
    """Load regression models from JSON files"""
    models = {}

    # Load core cost prediction model
    try:
        core_model_path = '../betler_cost_analysis/output/core_regression_model.json'
        if os.path.exists(core_model_path):
            with open(core_model_path, 'r') as f:
                models['core'] = json.load(f)
            print(f"Loaded core model: R² = {models['core']['model']['r_squared']:.4f}")
        else:
            print("Warning: Core prediction model not found. Run core analysis first.")
    except Exception as e:
        print(f"Error loading core prediction model: {e}")

    # Load Cognito prediction model
    try:
        cognito_model_path = '../cognito_cost_analysis/output/cognito_regression_model.json'
        if os.path.exists(cognito_model_path):
            with open(cognito_model_path, 'r') as f:
                models['cognito'] = json.load(f)
            print(f"Loaded Cognito model: R² = {models['cognito']['base_model']['r_squared']:.4f}")
        else:
            print("Warning: Cognito prediction model not found. Run Cognito analysis first.")
    except Exception as e:
        print(f"Error loading Cognito prediction model: {e}")

    return models

def extract_latest_metrics(core_data, cognito_data):
    """Extract the most recent metrics to use as baseline"""
    baseline = {}

    if 'df' in core_data and len(core_data['df']) > 0:
        core_df = core_data['df']
        # Get latest complete record
        latest_core = core_df.dropna().iloc[-1] if len(core_df.dropna()) > 0 else None
        if latest_core is not None:
            baseline['transactions'] = latest_core.get('daily_transaction_volume', 0)
            baseline['customers'] = latest_core.get('daily_customer_volume', 0)
            baseline['core_cost'] = latest_core.get('daily_cost', 0)
            baseline['date'] = latest_core.get('date', 'unknown')

    if 'df' in cognito_data and len(cognito_data['df']) > 0:
        cognito_df = cognito_data['df']
        # Get latest complete record
        latest_cognito = cognito_df.dropna().iloc[-1] if len(cognito_df.dropna()) > 0 else None
        if latest_cognito is not None:
            # Use monthly data as baseline, convert to daily approximation
            baseline['monthly_transactions'] = latest_cognito.get('monthly_transaction_volume', 0)
            baseline['monthly_customers'] = latest_cognito.get('monthly_customer_volume', 0)
            baseline['cognito_cost'] = latest_cognito.get('cognito_cost', 0)
            baseline['month'] = latest_cognito.get('month', 'unknown')

    return baseline

def generate_future_projections(baseline, models, months_ahead=12, transaction_growth_rate=0.05, customer_growth_rate=0.03):
    """Generate month-by-month projections using loaded regression models"""

    projections = []
    current_transactions = baseline.get('monthly_transactions', 0)
    current_customers = baseline.get('monthly_customers', 0)

    # Extract model parameters
    cognito_model = models.get('cognito', {})
    core_model = models.get('core', {})

    # Start from next month
    start_date = datetime.now().replace(day=1) + timedelta(days=32)
    start_date = start_date.replace(day=1)

    for month in range(months_ahead):
        # Calculate date
        projection_date = start_date + timedelta(days=month*30)

        # Apply compound growth
        projected_transactions = current_transactions * ((1 + transaction_growth_rate) ** month)
        projected_customers = current_customers * ((1 + customer_growth_rate) ** month)

        # Predict Cognito costs using the hybrid model
        predicted_cognito_cost = 0
        if cognito_model and 'base_model' in cognito_model:
            base_model = cognito_model['base_model']
            optimization_factor = cognito_model.get('optimization_factor', 0.4)

            # Use the regression equation: Cost = intercept + trans_coeff * transactions + cust_coeff * customers
            base_cost = (base_model['intercept'] +
                        base_model['transaction_coefficient'] * projected_transactions +
                        base_model['customer_coefficient'] * projected_customers)

            # Apply post-optimization factor (currently all predictions are post-July 2025)
            predicted_cognito_cost = base_cost * optimization_factor

        # Predict Core AWS costs using the core model
        predicted_core_cost = 0
        if core_model and 'model' in core_model:
            model = core_model['model']

            # Convert monthly projections to daily for core model (which was trained on daily data)
            daily_transactions = projected_transactions / 30  # Approximate daily from monthly
            daily_customers = projected_customers  # Customer count is same daily/monthly

            # Use the regression equation
            daily_cost = (model['intercept'] +
                         model['transaction_coefficient'] * daily_transactions +
                         model['customer_coefficient'] * daily_customers)

            # Convert back to monthly
            predicted_core_cost = daily_cost * 30

        # Total predicted cost
        total_predicted_cost = predicted_cognito_cost + predicted_core_cost

        projections.append({
            'date': projection_date.strftime('%Y-%m'),
            'month_offset': month + 1,
            'projected_transactions': projected_transactions,
            'projected_customers': projected_customers,
            'predicted_cognito_cost': predicted_cognito_cost,
            'predicted_core_cost': predicted_core_cost,
            'predicted_total_cost': total_predicted_cost,
            'transaction_growth_rate': transaction_growth_rate,
            'customer_growth_rate': customer_growth_rate
        })

    return projections

def create_predictive_dashboard(baseline, projections, output_dir='output'):
    """Create comprehensive predictive dashboard"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert projections to DataFrame for easier plotting
    proj_df = pd.DataFrame(projections)
    proj_df['date_obj'] = pd.to_datetime(proj_df['date'] + '-01')

    # Create dashboard with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Betler Platform: Predictive Cost Analysis Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Projected Customer Growth
    ax1 = axes[0, 0]
    ax1.plot(proj_df['date_obj'], proj_df['projected_customers'], 'g-', linewidth=2, marker='o')
    ax1.set_title('Projected Customer Growth', fontweight='bold')
    ax1.set_ylabel('Monthly Customers')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    # Add growth rate annotation
    growth_rate = projections[0]['customer_growth_rate'] * 100
    ax1.text(0.02, 0.98, f'Growth Rate: {growth_rate:.1f}%/month',
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Plot 2: Projected Transaction Growth
    ax2 = axes[0, 1]
    ax2.plot(proj_df['date_obj'], proj_df['projected_transactions'], 'b-', linewidth=2, marker='s')
    ax2.set_title('Projected Transaction Growth', fontweight='bold')
    ax2.set_ylabel('Monthly Transactions')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)

    # Add growth rate annotation
    growth_rate = projections[0]['transaction_growth_rate'] * 100
    ax2.text(0.02, 0.98, f'Growth Rate: {growth_rate:.1f}%/month',
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Plot 3: Cost Breakdown Over Time
    ax3 = axes[0, 2]
    ax3.plot(proj_df['date_obj'], proj_df['predicted_cognito_cost'], 'r-', linewidth=2, label='Cognito Costs')
    ax3.plot(proj_df['date_obj'], proj_df['predicted_core_cost'], 'purple', linewidth=2, label='Core AWS Costs')
    ax3.plot(proj_df['date_obj'], proj_df['predicted_total_cost'], 'k--', linewidth=3, label='Total Costs')
    ax3.set_title('Predicted Cost Breakdown', fontweight='bold')
    ax3.set_ylabel('Monthly Cost (USD)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Monthly Cost Growth
    ax4 = axes[1, 0]
    monthly_growth = [0] + [((projections[i]['predicted_total_cost'] / projections[i-1]['predicted_total_cost']) - 1) * 100
                           for i in range(1, len(projections))]
    ax4.bar(range(len(monthly_growth)), monthly_growth, alpha=0.7, color='orange')
    ax4.set_title('Month-over-Month Cost Growth', fontweight='bold')
    ax4.set_ylabel('Growth Rate (%)')
    ax4.set_xlabel('Month Offset')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Plot 5: Cost per Customer Trend
    ax5 = axes[1, 1]
    cost_per_customer = proj_df['predicted_total_cost'] / proj_df['projected_customers']
    ax5.plot(proj_df['date_obj'], cost_per_customer, 'green', linewidth=2, marker='d')
    ax5.set_title('Cost per Customer Trend', fontweight='bold')
    ax5.set_ylabel('Cost per Customer (USD)')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax5.tick_params(axis='x', rotation=45)

    # Plot 6: Annual Cost Projection Summary
    ax6 = axes[1, 2]
    # Show first 12 months as bars
    months_to_show = min(12, len(projections))
    bar_data = projections[:months_to_show]
    x_pos = range(months_to_show)

    cognito_costs = [p['predicted_cognito_cost'] for p in bar_data]
    core_costs = [p['predicted_core_cost'] for p in bar_data]

    ax6.bar(x_pos, cognito_costs, label='Cognito', alpha=0.8, color='red')
    ax6.bar(x_pos, core_costs, bottom=cognito_costs, label='Core AWS', alpha=0.8, color='purple')

    ax6.set_title('12-Month Cost Projection', fontweight='bold')
    ax6.set_ylabel('Monthly Cost (USD)')
    ax6.set_xlabel('Month')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add total annual cost annotation
    annual_total = sum([p['predicted_total_cost'] for p in bar_data])
    ax6.text(0.02, 0.98, f'Annual Total: ${annual_total:,.0f}',
             transform=ax6.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    plt.tight_layout()

    # Save dashboard
    dashboard_path = f'{output_dir}/predictive_cost_dashboard.png'
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"Predictive dashboard saved: {dashboard_path}")

    return fig

def save_projections_csv(projections, baseline, output_dir='output'):
    """Save projections to CSV file"""

    # Create comprehensive CSV with all projection data
    proj_df = pd.DataFrame(projections)

    # Add baseline information as metadata
    metadata_rows = []
    metadata_rows.append(['# BETLER PREDICTIVE COST ANALYSIS'])
    metadata_rows.append(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    metadata_rows.append(['# Baseline Date:', baseline.get('month', 'unknown')])
    metadata_rows.append(['# Baseline Customers:', baseline.get('monthly_customers', 0)])
    metadata_rows.append(['# Baseline Transactions:', baseline.get('monthly_transactions', 0)])
    metadata_rows.append(['# Baseline Cognito Cost:', baseline.get('cognito_cost', 0)])
    metadata_rows.append([''])

    # Save CSV
    csv_path = f'{output_dir}/predictive_cost_analysis.csv'

    # Write metadata first
    with open(csv_path, 'w') as f:
        for row in metadata_rows:
            f.write(','.join(map(str, row)) + '\n')

    # Append DataFrame
    proj_df.to_csv(csv_path, mode='a', index=False)
    print(f"Projections saved: {csv_path}")

    return csv_path

def create_summary_report(baseline, projections, output_dir='output'):
    """Generate text summary report"""

    report_path = f'{output_dir}/predictive_analysis_summary.txt'

    with open(report_path, 'w') as f:
        f.write("BETLER PREDICTIVE COST ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Baseline Period: {baseline.get('month', 'unknown')}\n")
        f.write(f"Projection Period: {len(projections)} months\n\n")

        f.write("BASELINE METRICS:\n")
        f.write(f"  Monthly Customers: {baseline.get('monthly_customers', 0):,.0f}\n")
        f.write(f"  Monthly Transactions: {baseline.get('monthly_transactions', 0):,.0f}\n")
        f.write(f"  Monthly Cognito Cost: ${baseline.get('cognito_cost', 0):,.2f}\n\n")

        f.write("GROWTH PARAMETERS:\n")
        f.write(f"  Customer Growth Rate: {projections[0]['customer_growth_rate']*100:.1f}% per month\n")
        f.write(f"  Transaction Growth Rate: {projections[0]['transaction_growth_rate']*100:.1f}% per month\n\n")

        # Calculate key summary metrics
        first_month = projections[0]
        last_month = projections[-1]
        annual_projections = projections[:12] if len(projections) >= 12 else projections

        f.write("12-MONTH PROJECTIONS:\n")
        f.write(f"  Customers (Month 1): {first_month['projected_customers']:,.0f}\n")
        f.write(f"  Customers (Month 12): {projections[11]['projected_customers']:,.0f}\n" if len(projections) >= 12 else "")
        f.write(f"  Customer Growth: {((projections[11]['projected_customers'] / first_month['projected_customers']) - 1) * 100:.1f}%\n" if len(projections) >= 12 else "")
        f.write("\n")

        f.write(f"  Total Cost (Month 1): ${first_month['predicted_total_cost']:,.2f}\n")
        f.write(f"  Total Cost (Month 12): ${projections[11]['predicted_total_cost']:,.2f}\n" if len(projections) >= 12 else "")
        f.write(f"  Annual Total Cost: ${sum([p['predicted_total_cost'] for p in annual_projections]):,.2f}\n")
        f.write("\n")

        f.write("COST BREAKDOWN (12-Month Total):\n")
        annual_cognito = sum([p['predicted_cognito_cost'] for p in annual_projections])
        annual_core = sum([p['predicted_core_cost'] for p in annual_projections])
        annual_total = annual_cognito + annual_core

        f.write(f"  Cognito Costs: ${annual_cognito:,.2f} ({annual_cognito/annual_total*100:.1f}%)\n")
        f.write(f"  Core AWS Costs: ${annual_core:,.2f} ({annual_core/annual_total*100:.1f}%)\n")
        f.write(f"  Total: ${annual_total:,.2f}\n")

    print(f"Summary report saved: {report_path}")
    return report_path

def main():
    """Main execution function with command line arguments"""

    parser = argparse.ArgumentParser(description='Betler Predictive Cost Analysis')
    parser.add_argument('--customer-growth', type=float, default=0.03,
                       help='Monthly customer growth rate (default: 0.03 = 3%%)')
    parser.add_argument('--transaction-growth', type=float, default=0.05,
                       help='Monthly transaction growth rate (default: 0.05 = 5%%)')
    parser.add_argument('--months', type=int, default=12,
                       help='Number of months to project (default: 12)')

    args = parser.parse_args()

    print("=" * 60)
    print("BETLER PREDICTIVE COST ANALYSIS")
    print("=" * 60)
    print(f"Customer Growth Rate: {args.customer_growth*100:.1f}% per month")
    print(f"Transaction Growth Rate: {args.transaction_growth*100:.1f}% per month")
    print(f"Projection Period: {args.months} months")
    print("")

    # Load existing analysis data
    print("Loading existing analysis data...")
    core_data, cognito_data = load_existing_analysis_data()

    if not core_data and not cognito_data:
        print("Error: No analysis data found. Please run the main analysis scripts first.")
        sys.exit(1)

    # Load prediction models
    print("Loading prediction models...")
    models = load_prediction_models()

    if not models:
        print("Error: No prediction models found. Please run the main analysis scripts first.")
        sys.exit(1)

    # Extract baseline metrics
    print("Extracting baseline metrics...")
    baseline = extract_latest_metrics(core_data, cognito_data)

    if not baseline:
        print("Error: Could not extract baseline metrics from analysis data.")
        sys.exit(1)

    print(f"Baseline: {baseline.get('monthly_customers', 0):,.0f} customers, "
          f"{baseline.get('monthly_transactions', 0):,.0f} transactions/month")

    # Generate projections
    print("Generating cost projections...")
    projections = generate_future_projections(
        baseline,
        models,
        months_ahead=args.months,
        transaction_growth_rate=args.transaction_growth,
        customer_growth_rate=args.customer_growth
    )

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Create dashboard
    print("Creating predictive dashboard...")
    create_predictive_dashboard(baseline, projections, output_dir)

    # Save projections
    print("Saving projection data...")
    save_projections_csv(projections, baseline, output_dir)

    # Create summary report
    print("Generating summary report...")
    create_summary_report(baseline, projections, output_dir)

    print("\n" + "=" * 60)
    print("PREDICTIVE ANALYSIS COMPLETE")
    print("=" * 60)
    print("Files created:")
    print(f"  - {output_dir}/predictive_cost_dashboard.png")
    print(f"  - {output_dir}/predictive_cost_analysis.csv")
    print(f"  - {output_dir}/predictive_analysis_summary.txt")
    print("")
    print("Key Predictions:")
    annual_total = sum([p['predicted_total_cost'] for p in projections[:12]])
    print(f"  Annual Cost Projection: ${annual_total:,.2f}")
    final_customers = projections[11]['projected_customers'] if len(projections) >= 12 else projections[-1]['projected_customers']
    print(f"  Year-End Customers: {final_customers:,.0f}")

if __name__ == "__main__":
    main()