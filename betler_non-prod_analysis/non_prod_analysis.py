#!/usr/bin/env python3

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
# Simple linear regression without sklearn
import os

def load_and_process_costs():
    """Load AWS cost data and process by environment."""

    # Load environment-grouped costs
    with open('output/aws_costs_by_environment.json', 'r') as f:
        env_data = json.load(f)

    # Load overall daily costs
    with open('output/aws_daily_costs.json', 'r') as f:
        daily_data = json.load(f)


    # Process environment data
    env_costs = []
    for result in env_data['ResultsByTime']:
        date = result['TimePeriod']['Start']
        for group in result['Groups']:
            if group['Keys'] and group['Keys'][0]:  # Skip empty environment tags
                env_name = group['Keys'][0]
                cost = float(group['Metrics']['AmortizedCost']['Amount'])
                env_costs.append({
                    'date': date,
                    'environment': env_name,
                    'cost': cost
                })

    # Process daily total costs
    daily_costs = []
    for result in daily_data['ResultsByTime']:
        date = result['TimePeriod']['Start']
        cost = float(result['Total']['AmortizedCost']['Amount'])
        daily_costs.append({
            'date': date,
            'cost': cost
        })

    env_df = pd.DataFrame(env_costs)
    daily_df = pd.DataFrame(daily_costs)

    # Convert dates
    env_df['date'] = pd.to_datetime(env_df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    return env_df, daily_df

def create_environment_dashboard(env_df, daily_df, predictions=None):
    """Create visualization dashboard showing cost per environment over 12 months."""

    # Define persistent vs transient environments (accounting for environment$ prefix)
    persistent_envs = ['dev01', 'qa', 'stage']

    # Categorize environments - check if the environment name ends with any persistent env
    def categorize_env(env_name):
        # Remove environment$ prefix and any trailing domain (.betler)
        clean_name = env_name.replace('environment$', '').split('.')[0]
        return 'Persistent' if clean_name in persistent_envs else 'Transient'

    env_df['env_type'] = env_df['environment'].apply(categorize_env)

    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Non-Production Environment Cost Analysis - Last 12 Months', fontsize=16, fontweight='bold')

    # Create 3x2 grid with 6th panel for projections
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Full width for stacked chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # 1. Cost per environment over time (stacked area chart)
    env_pivot = env_df.pivot(index='date', columns='environment', values='cost').fillna(0)

    # Find top 8 environments by total cost
    env_totals = env_pivot.sum().sort_values(ascending=False)
    top_8_envs = env_totals.head(8).index.tolist()
    other_envs = [env for env in env_pivot.columns if env not in top_8_envs]

    # Create new dataframe with top 8 + "Remaining"
    plot_data = env_pivot[top_8_envs].copy()
    if other_envs:
        plot_data['Remaining'] = env_pivot[other_envs].sum(axis=1)

    labels = list(plot_data.columns)
    ax1.stackplot(plot_data.index, *[plot_data[col] for col in plot_data.columns],
                  labels=labels, alpha=0.7)
    ax1.set_title('Daily Cost per Environment (Stacked)', fontweight='bold')
    ax1.set_ylabel('Cost ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Persistent vs Transient costs over time
    persistent_daily = env_df[env_df['env_type'] == 'Persistent'].groupby('date')['cost'].sum().reindex(daily_df['date'], fill_value=0)
    transient_daily = env_df[env_df['env_type'] == 'Transient'].groupby('date')['cost'].sum().reindex(daily_df['date'], fill_value=0)

    ax2.fill_between(daily_df['date'], 0, persistent_daily, alpha=0.7, label='Persistent Environments', color='steelblue')
    ax2.fill_between(daily_df['date'], persistent_daily, persistent_daily + transient_daily, alpha=0.7, label='Transient Environments', color='lightcoral')

    ax2.set_title('Daily Costs: Persistent vs Transient Environments', fontweight='bold')
    ax2.set_ylabel('Cost ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Persistent vs Transient distribution (pie chart)
    recent_env = env_df[env_df['date'] >= env_df['date'].max() - timedelta(days=30)]
    env_type_totals = recent_env.groupby('env_type')['cost'].sum()

    ax3.pie(env_type_totals.values, labels=env_type_totals.index, autopct='%1.1f%%', startangle=90,
            colors=['steelblue', 'lightcoral'])
    ax3.set_title('Cost Distribution: Persistent vs Transient (Last 30 Days)', fontweight='bold')

    # 4. Persistent environments monthly trends
    env_df['month'] = env_df['date'].dt.to_period('M')
    persistent_envs_data = env_df[env_df['env_type'] == 'Persistent']
    monthly_persistent = persistent_envs_data.groupby(['month', 'environment'])['cost'].sum().reset_index()
    monthly_persistent_pivot = monthly_persistent.pivot(index='month', columns='environment', values='cost').fillna(0)

    # Plot persistent environments (need to match actual environment names)
    for col in monthly_persistent_pivot.columns:
        clean_name = col.replace('environment$', '').split('.')[0]
        if clean_name in persistent_envs:
            ax4.plot(monthly_persistent_pivot.index.astype(str), monthly_persistent_pivot[col],
                    marker='o', label=clean_name, linewidth=2)

    ax4.set_title('Monthly Cost Trends - Persistent Environments', fontweight='bold')
    ax4.set_ylabel('Monthly Cost ($)')
    ax4.set_xlabel('Month')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # 5. Transient environments activity count over time
    transient_envs_data = env_df[env_df['env_type'] == 'Transient']
    monthly_transient_activity = transient_envs_data.groupby([
        transient_envs_data['date'].dt.to_period('M')
    ])['environment'].nunique().reset_index()
    monthly_transient_activity['month_str'] = monthly_transient_activity['date'].astype(str)

    ax5.bar(monthly_transient_activity['month_str'], monthly_transient_activity['environment'],
           color='lightcoral', alpha=0.7)
    ax5.set_title('Transient Environments - Active Count by Month', fontweight='bold')
    ax5.set_ylabel('Number of Active Environments')
    ax5.set_xlabel('Month')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

    # 6. Add 12-month projection if predictions available
    if predictions and 'combined' in predictions:
        # Add a new subplot at the bottom for projections
        gs_proj = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
        # Move existing plots
        ax1.set_position(gs_proj[0, :].get_position(fig))
        ax2.set_position(gs_proj[1, 0].get_position(fig))
        ax3.set_position(gs_proj[1, 1].get_position(fig))
        ax4.set_position(gs_proj[2, 0].get_position(fig))
        ax5.set_position(gs_proj[2, 1].get_position(fig))

        # Add projection chart
        ax6 = fig.add_subplot(gs_proj[3, :])

        months = [f"Month {p['month']}" for p in predictions['combined']['monthly_projection']]
        persistent_costs = [p['persistent_cost'] for p in predictions['combined']['monthly_projection']]
        transient_costs = [p['transient_cost'] for p in predictions['combined']['monthly_projection']]

        ax6.bar(months, persistent_costs, alpha=0.7, label='Persistent Environments', color='steelblue')
        ax6.bar(months, transient_costs, bottom=persistent_costs, alpha=0.7, label='Transient Environments', color='lightcoral')

        ax6.set_title('12-Month Cost Projection: Persistent vs Transient', fontweight='bold')
        ax6.set_ylabel('Monthly Cost ($)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)

        # Add totals text
        total_12_months = predictions['combined']['total_12_months']
        persistent_total = predictions['persistent']['next_12_months_total'] if 'persistent' in predictions else 0
        transient_total = predictions['transient']['next_12_months_total'] if 'transient' in predictions else 0

        ax6.text(0.02, 0.98, f'12-Month Total: ${total_12_months:,.0f}\nPersistent: ${persistent_total:,.0f}\nTransient: ${transient_total:,.0f}',
                transform=ax6.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('output/non_prod_environment_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def simple_linear_regression(x, y):
    """Simple linear regression using numpy."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        slope = 0
        intercept = y_mean
        r2 = 0
        rmse = 0
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        # Debug R² calculation
        print(f"    DEBUG R² calc: ss_res={ss_res:.2f}, ss_tot={ss_tot:.2f}")

        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    return slope, intercept, r2, rmse

def predict_environment_costs(env_df, daily_df):
    """Create 12-month predictions with separate models for persistent vs transient."""

    # Categorize environments
    persistent_envs_list = ['dev01', 'qa', 'stage']

    def categorize_env(env_name):
        clean_name = env_name.replace('environment$', '').split('.')[0]
        return 'Persistent' if clean_name in persistent_envs_list else 'Transient'

    env_df['env_type'] = env_df['environment'].apply(categorize_env)

    predictions = {
        'methodology': {
            'persistent': 'Linear regression based on historical growth trends',
            'transient': 'Monthly budget cap = max_historical_monthly_cost / 3'
        }
    }

    # PERSISTENT ENVIRONMENTS PREDICTION (Linear Regression)
    persistent_data = env_df[env_df['env_type'] == 'Persistent'].copy()

    if len(persistent_data) > 0:
        # Aggregate persistent costs by day
        persistent_daily = persistent_data.groupby('date')['cost'].sum().reset_index()
        persistent_daily['days_since_start'] = (persistent_daily['date'] - persistent_daily['date'].min()).dt.days

        x = persistent_daily['days_since_start'].values
        y = persistent_daily['cost'].values

        slope, intercept, r2, rmse = simple_linear_regression(x, y)

        # Debug output to understand R² calculation
        print(f"DEBUG - Persistent environments regression (daily data):")
        print(f"  Data points: {len(x)} days")
        print(f"  Date range: {persistent_daily['date'].min()} to {persistent_daily['date'].max()}")
        print(f"  Daily cost range: ${y.min():.2f} to ${y.max():.2f}")
        print(f"  Mean daily cost: ${y.mean():.2f}")
        print(f"  Slope: ${slope:.4f}/day")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: ${rmse:.2f}")

        # Check for data quality issues
        zero_days = sum(y == 0)
        print(f"  Zero-cost days: {zero_days}/{len(y)} ({zero_days/len(y)*100:.1f}%)")

        # Predict next 12 months (365 days)
        future_days = np.arange(persistent_daily['days_since_start'].max() + 1,
                               persistent_daily['days_since_start'].max() + 366)
        future_daily_costs = slope * future_days + intercept
        # Ensure no negative predictions
        future_daily_costs = np.maximum(future_daily_costs, 0)

        # Calculate total for 12 months
        total_12_months = float(future_daily_costs.sum())

        predictions['persistent'] = {
            'model_type': 'linear_regression_daily',
            'model_r2': float(r2),
            'model_rmse': float(rmse),
            'daily_trend': float(slope),
            'current_daily_avg': float(persistent_daily['cost'].tail(30).mean()),
            'next_12_months_total': total_12_months,
            'monthly_projection': []
        }

        # Monthly breakdown for persistent (approximately 30.4 days per month)
        days_per_month = 365 / 12
        for month in range(12):
            start_day = int(month * days_per_month)
            end_day = int((month + 1) * days_per_month)
            # Handle edge case for last month
            if month == 11:
                end_day = len(future_daily_costs)
            month_daily_costs = future_daily_costs[start_day:end_day]
            predictions['persistent']['monthly_projection'].append({
                'month': month + 1,
                'cost': float(month_daily_costs.sum())
            })

    # TRANSIENT ENVIRONMENTS PREDICTION (Budget Cap)
    transient_data = env_df[env_df['env_type'] == 'Transient'].copy()

    if len(transient_data) > 0:
        # Calculate monthly transient costs
        transient_data['month'] = transient_data['date'].dt.to_period('M')
        monthly_transient = transient_data.groupby('month')['cost'].sum()

        max_monthly_cost = float(monthly_transient.max())

        # New budget model: $1600/day = $48,800/month
        # Based on $400/day per environment × 2-4 environments
        daily_budget = 1600.0  # $400/day per env × 4 max environments
        monthly_budget_cap = daily_budget * 30.4  # Average days per month

        predictions['transient'] = {
            'model_type': 'daily_budget_cap',
            'max_historical_monthly': max_monthly_cost,
            'daily_budget': daily_budget,
            'monthly_budget_cap': monthly_budget_cap,
            'reasoning': '$400/day per environment × 2-4 environments = $1600/day budget',
            'note': 'High variability area under active development - predictions should be revisited',
            'next_12_months_total': monthly_budget_cap * 12,
            'monthly_projection': []
        }

        # Monthly breakdown for transient (same cap each month)
        for month in range(12):
            predictions['transient']['monthly_projection'].append({
                'month': month + 1,
                'cost': monthly_budget_cap
            })

    # COMBINED PROJECTION
    total_12_months = 0
    if 'persistent' in predictions:
        total_12_months += predictions['persistent']['next_12_months_total']
    if 'transient' in predictions:
        total_12_months += predictions['transient']['next_12_months_total']

    predictions['combined'] = {
        'total_12_months': total_12_months,
        'monthly_projection': []
    }

    # Combined monthly breakdown
    for month in range(12):
        persistent_cost = predictions['persistent']['monthly_projection'][month]['cost'] if 'persistent' in predictions else 0
        transient_cost = predictions['transient']['monthly_projection'][month]['cost'] if 'transient' in predictions else 0

        predictions['combined']['monthly_projection'].append({
            'month': month + 1,
            'persistent_cost': persistent_cost,
            'transient_cost': transient_cost,
            'total_cost': persistent_cost + transient_cost
        })

    return predictions

def create_results_json(predictions, env_df, daily_df):
    """Create structured results JSON for integration."""

    results = {
        "analysis_type": "non_prod_environment_analysis",
        "analysis_date": datetime.now().isoformat(),
        "data_period": {
            "start": daily_df['date'].min().strftime('%Y-%m-%d'),
            "end": daily_df['date'].max().strftime('%Y-%m-%d'),
            "days": len(daily_df)
        },
        "overall_metrics": {
            "total_cost_12_months": float(daily_df['cost'].sum()),
            "average_daily_cost": float(daily_df['cost'].mean()),
            "environments_found": list(env_df['environment'].unique())
        },
        "predictions": predictions
    }

    with open('output/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    print("Processing non-prod environment cost data...")

    # Load and process data
    env_df, daily_df = load_and_process_costs()

    # Get persistent vs transient counts
    persistent_envs = ['dev01', 'qa', 'stage']
    all_envs = set(env_df['environment'].unique())

    # Parse environment names to find matches
    def parse_env_name(env_name):
        return env_name.replace('environment$', '').split('.')[0]

    found_persistent = []
    transient_envs = []

    for env in all_envs:
        clean_name = parse_env_name(env)
        if clean_name in persistent_envs:
            found_persistent.append(env)
        else:
            transient_envs.append(env)

    print(f"✓ Processed {len(daily_df)} days of cost data")
    print(f"✓ Found {len(all_envs)} total environments:")
    print(f"  - Persistent: {len(found_persistent)} ({', '.join(found_persistent)})")
    print(f"  - Transient: {len(transient_envs)} ({', '.join(transient_envs)})")

    # Generate predictions first
    print("Generating cost predictions...")
    predictions = predict_environment_costs(env_df, daily_df)
    print(f"✓ Generated predictions")

    # Create dashboard with predictions
    print("Creating environment cost dashboard...")
    create_environment_dashboard(env_df, daily_df, predictions)
    print("✓ Dashboard saved: output/non_prod_environment_dashboard.png")

    # Create results
    results = create_results_json(predictions, env_df, daily_df)
    print("✓ Results saved: output/results.json")

    # Enhanced Summary with 12-month projection
    print("\n" + "="*70)
    print("NON-PROD ENVIRONMENT COST ANALYSIS & 12-MONTH PROJECTION")
    print("="*70)

    # Historical summary
    print(f"📊 Historical (12 months): ${results['overall_metrics']['total_cost_12_months']:,.2f}")
    print(f"📈 Average daily cost: ${results['overall_metrics']['average_daily_cost']:,.2f}")

    # 12-month projection
    if 'combined' in predictions:
        print(f"\n🎯 12-MONTH PROJECTION: ${predictions['combined']['total_12_months']:,.2f}")

        if 'persistent' in predictions:
            print(f"  🏗️  Persistent (dev01, qa, stage): ${predictions['persistent']['next_12_months_total']:,.2f}")
            print(f"      • Model: Linear regression (R² {predictions['persistent']['model_r2']:.3f})")
            print(f"      • Daily trend: ${predictions['persistent']['daily_trend']:+.2f}/day")

        if 'transient' in predictions:
            print(f"  ⚡ Transient (load testing, etc): ${predictions['transient']['next_12_months_total']:,.2f}")
            print(f"      • Model: Budget cap (${predictions['transient']['monthly_budget_cap']:,.0f}/month)")
            print(f"      • Based on: Max historical (${predictions['transient']['max_historical_monthly']:,.0f}) ÷ 3")

        # Monthly breakdown for first 3 months
        print(f"\n📅 NEXT 3 MONTHS BREAKDOWN:")
        for i in range(3):
            month_data = predictions['combined']['monthly_projection'][i]
            print(f"  Month {month_data['month']}: ${month_data['total_cost']:,.0f} "
                  f"(Persistent: ${month_data['persistent_cost']:,.0f}, "
                  f"Transient: ${month_data['transient_cost']:,.0f})")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()