#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import sys
import os

def load_and_prepare_monthly_data():
    """Load monthly CSV data and prepare for plotting"""
    try:
        df = pd.read_csv('output/cognito_cost_analysis.csv')

        # Convert month column to datetime (first day of month)
        df['date'] = pd.to_datetime(df['month'] + '-01')

        # Remove rows with missing data for cleaner plots
        df_clean = df.dropna()

        print(f"Loaded {len(df)} total months, {len(df_clean)} months with complete data")
        return df, df_clean
    except FileNotFoundError:
        print("Error: cognito_cost_analysis.csv not found. Please run the main analysis script first.")
        sys.exit(1)

def create_monthly_plots(df, df_clean):
    """Create comprehensive monthly visualization plots"""

    # Create figure with subplots - 3x2 grid (2 wide, 3 high)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Betler Platform: Monthly Cognito Cost Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Monthly Transaction Volume
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['monthly_transaction_volume'], color='blue', linewidth=2, marker='o', alpha=0.7)
    ax1.set_title('Monthly Transaction Volume', fontweight='bold')
    ax1.set_ylabel('Monthly Transactions')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    # Add trend line
    if len(df.dropna(subset=['monthly_transaction_volume'])) > 1:
        x_num = mdates.date2num(df.dropna(subset=['monthly_transaction_volume'])['date'])
        y_vals = df.dropna(subset=['monthly_transaction_volume'])['monthly_transaction_volume']
        z = np.polyfit(x_num, y_vals, 1)
        p = np.poly1d(z)
        ax1.plot(df.dropna(subset=['monthly_transaction_volume'])['date'], p(x_num),
                "r--", alpha=0.8, linewidth=2, label='Trend')
        ax1.legend()

    # Plot 2: Monthly Customer Volume
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['monthly_customer_volume'], color='green', linewidth=2, marker='o', alpha=0.7)
    ax2.set_title('Monthly Customer Volume', fontweight='bold')
    ax2.set_ylabel('Monthly Customer Volume')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)

    # Add trend line
    if len(df.dropna(subset=['monthly_customer_volume'])) > 1:
        x_num = mdates.date2num(df.dropna(subset=['monthly_customer_volume'])['date'])
        y_vals = df.dropna(subset=['monthly_customer_volume'])['monthly_customer_volume']
        z = np.polyfit(x_num, y_vals, 1)
        p = np.poly1d(z)
        ax2.plot(df.dropna(subset=['monthly_customer_volume'])['date'], p(x_num),
                "r--", alpha=0.8, linewidth=2, label='Trend')
        ax2.legend()

    # Plot 3: Dual-Axis Customer Volume vs Cognito Costs
    ax3 = axes[1, 0]

    # Left y-axis: Customer Volume
    color = 'tab:green'
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Monthly Customer Volume', color=color)
    line1 = ax3.plot(df['date'], df['monthly_customer_volume'], color=color, linewidth=2, marker='s', alpha=0.7, label='Customer Volume')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45)

    # Create second y-axis for Cognito costs
    ax3_twin = ax3.twinx()
    color = 'tab:red'
    ax3_twin.set_ylabel('Monthly Cognito Cost (USD)', color=color)
    if len(df_clean) > 0:
        line2 = ax3_twin.plot(df_clean['date'], df_clean['cognito_cost'], color=color, linewidth=2, marker='o', alpha=0.7, label='Cognito Cost')
        ax3_twin.tick_params(axis='y', labelcolor=color)

        # Add title and combined legend
        ax3.set_title('Customer Volume vs Cognito Costs', fontweight='bold')
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax3.set_title('Customer Volume vs Cognito Costs (No Cost Data)', fontweight='bold')

    # Plot 4: Dual-Axis Transaction Volume vs Cognito Costs
    ax4 = axes[1, 1]

    # Left y-axis: Transaction Volume
    color = 'tab:blue'
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Monthly Transaction Volume', color=color)
    line1 = ax4.plot(df['date'], df['monthly_transaction_volume'], color=color, linewidth=2, marker='o', alpha=0.7, label='Transaction Volume')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.tick_params(axis='x', rotation=45)

    # Create second y-axis for Cognito costs
    ax4_twin = ax4.twinx()
    color = 'tab:red'
    ax4_twin.set_ylabel('Monthly Cognito Cost (USD)', color=color)
    if len(df_clean) > 0:
        line2 = ax4_twin.plot(df_clean['date'], df_clean['cognito_cost'], color=color, linewidth=2, marker='s', alpha=0.7, label='Cognito Cost')
        ax4_twin.tick_params(axis='y', labelcolor=color)

        # Add title and combined legend
        ax4.set_title('Transaction Volume vs Cognito Costs', fontweight='bold')
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax4.set_title('Transaction Volume vs Cognito Costs (No Cost Data)', fontweight='bold')

    # Plot 5: Monthly Cognito Costs with Predictions
    ax5 = axes[2, 0]
    if len(df_clean) > 0:
        ax5.plot(df_clean['date'], df_clean['cognito_cost'], color='red', linewidth=2, marker='o', alpha=0.7,
                label='Actual Cognito Cost')
        ax5.set_title('Monthly Cognito Costs', fontweight='bold')
        ax5.set_ylabel('Cost (USD)')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.tick_params(axis='x', rotation=45)

        # Add predicted cost lines using both models
        try:
            # Separate pre and post optimization data
            df_pre_opt = df_clean[df_clean['date'] < '2025-07-01']

            if len(df_pre_opt) >= 3:
                # Train on pre-optimization data
                X_train = np.column_stack([df_pre_opt['monthly_transaction_volume'].values,
                                         df_pre_opt['monthly_customer_volume'].values])
                y_train = df_pre_opt['cognito_cost'].values

                X_train_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
                theta = np.linalg.solve(X_train_intercept.T @ X_train_intercept, X_train_intercept.T @ y_train)

                # Generate base predictions for all data using pre-optimization model
                X_all = np.column_stack([df_clean['monthly_transaction_volume'].values,
                                       df_clean['monthly_customer_volume'].values])
                X_all_intercept = np.column_stack([np.ones(X_all.shape[0]), X_all])
                base_predictions = X_all_intercept @ theta

                # Plot pre-optimization model predictions (unoptimized)
                ax5.plot(df_clean['date'], base_predictions, color='green', linestyle=':', linewidth=2, alpha=0.8,
                        label='Pre-Optimization Model')

                # Create hybrid predictions with different factors pre/post July
                hybrid_predictions = []
                cutoff_date = pd.to_datetime('2025-07-01')
                for i, date in enumerate(df_clean['date']):
                    if pd.to_datetime(date) < cutoff_date:
                        # Pre-optimization: use original predictions
                        hybrid_predictions.append(base_predictions[i])
                    else:
                        # Post-optimization: apply 60% reduction (multiply by 0.4)
                        hybrid_predictions.append(base_predictions[i] * 0.4)

                # Plot hybrid model predictions (optimized)
                ax5.plot(df_clean['date'], hybrid_predictions, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                        label='Hybrid Model (60% reduction July+)')
                ax5.legend()
            else:
                # Fallback to simple regression if insufficient pre-opt data
                X = np.column_stack([df_clean['monthly_transaction_volume'].values, df_clean['monthly_customer_volume'].values])
                y = df_clean['cognito_cost'].values
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
                predictions = X_with_intercept @ theta
                ax5.plot(df_clean['date'], predictions, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                        label='Simple Regression')
                ax5.legend()
        except Exception as e:
            # If prediction fails, just show the actual costs
            print(f"Warning: Could not generate predictions for plot: {e}")
            pass
    else:
        ax5.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Monthly Cognito Costs (No Data)', fontweight='bold')

    # Layout is now:
    # Row 0: [Transaction Volume] [Customer Volume]
    # Row 1: [Customer vs Cost (dual)] [Transaction vs Cost (dual)]
    # Row 2: [Cognito Costs with Predictions] [Hidden]
    axes[2, 1].set_visible(False)  # Hide bottom-right

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(f'{output_dir}/cognito_cost_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Cognito dashboard saved as: {output_dir}/cognito_cost_analysis_dashboard.png")

    return fig

def create_summary_stats(df, df_clean):
    """Generate and save summary statistics"""

    print("\\n" + "="*60)
    print("COGNITO COST ANALYSIS SUMMARY STATISTICS")
    print("="*60)

    # Date range
    print(f"Data Period: {df['month'].min()} to {df['month'].max()}")
    print(f"Total Months: {len(df)}")
    print(f"Months with Complete Data: {len(df_clean)}")

    # Transaction volume stats
    if not df['monthly_transaction_volume'].isna().all():
        trans_data = df['monthly_transaction_volume'].dropna()
        print(f"\\nMonthly Transaction Volume:")
        print(f"  Average: {trans_data.mean():,.0f} transactions/month")
        print(f"  Median:  {trans_data.median():,.0f} transactions/month")
        print(f"  Min:     {trans_data.min():,.0f} transactions/month")
        print(f"  Max:     {trans_data.max():,.0f} transactions/month")
        if len(trans_data) > 1:
            print(f"  Growth:  {((trans_data.iloc[-1] / trans_data.iloc[0]) - 1) * 100:.1f}% over period")

    # Customer volume stats
    if not df['monthly_customer_volume'].isna().all():
        cust_data = df['monthly_customer_volume'].dropna()
        print(f"\\nMonthly Customer Volume:")
        print(f"  Average: {cust_data.mean():,.0f}")
        print(f"  Median:  {cust_data.median():,.0f}")
        print(f"  Min:     {cust_data.min():,.0f}")
        print(f"  Max:     {cust_data.max():,.0f}")
        if len(cust_data) > 1:
            print(f"  Growth:  {((cust_data.iloc[-1] / cust_data.iloc[0]) - 1) * 100:.1f}% over period")

    # Cognito cost stats
    if len(df_clean) > 0:
        cost_data = df_clean['cognito_cost']
        print(f"\\nMonthly Cognito Costs:")
        print(f"  Average: ${cost_data.mean():,.2f}/month")
        print(f"  Median:  ${cost_data.median():,.2f}/month")
        print(f"  Min:     ${cost_data.min():,.2f}/month")
        print(f"  Max:     ${cost_data.max():,.2f}/month")
        print(f"  Total:   ${cost_data.sum():,.2f} (for {len(cost_data)} months)")

        # Annual projection
        annual_avg = cost_data.mean() * 12
        print(f"  Annual Projection: ${annual_avg:,.2f}/year")


def main():
    """Main execution function"""
    print("Loading monthly data...")
    df, df_clean = load_and_prepare_monthly_data()

    print("Creating visualization dashboard...")
    fig = create_monthly_plots(df, df_clean)

    print("Generating summary statistics...")
    create_summary_stats(df, df_clean)

    print("\\n" + "="*60)
    print("COGNITO VISUALIZATION COMPLETE")
    print("="*60)
    print("Files created:")
    print("  - output/cognito_cost_analysis_dashboard.png")
    print("\\nAnalysis shows monthly Cognito cost patterns and correlations.")

if __name__ == "__main__":
    main()