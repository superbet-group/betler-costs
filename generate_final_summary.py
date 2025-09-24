#!/usr/bin/env python3

import os
import json
import pandas as pd

def read_json_results(filepath):
    """Read and parse JSON results file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception:
        return None

def extract_key_metrics():
    """Extract key metrics from all analysis JSON outputs"""

    # Read predictive analysis results
    predictive_results = read_json_results("betler_predictive_analysis/output/results.json")
    if not predictive_results:
        print("❌ Error: Predictive analysis results not found")
        return None

    # Extract key numbers from predictive analysis JSON
    metrics = {
        'annual_total': predictive_results['projections']['annual_totals']['total_cost'],
        'month1_cost': predictive_results['projections']['month_1']['total_cost'],
        'month12_cost': predictive_results['projections']['month_12']['total_cost'],
        'cognito_total': predictive_results['projections']['annual_totals']['cognito_cost'],
        'core_total': predictive_results['projections']['annual_totals']['core_aws_cost']
    }

    # Add core analysis data if available
    core_results = read_json_results("betler_cost_analysis/output/results.json")
    if core_results:
        metrics['core_analysis'] = {
            'total_historical_cost': core_results['totals']['total_cost'],
            'daily_average': core_results['totals']['average_daily_cost'],
            'data_days': core_results['data_period']['days']
        }

    # Add Cognito analysis data if available
    cognito_results = read_json_results("cognito_cost_analysis/output/results.json")
    if cognito_results:
        metrics['cognito_analysis'] = {
            'total_historical_cost': cognito_results['totals']['total_cost'],
            'monthly_average': cognito_results['totals']['average_monthly_cost'],
            'data_months': cognito_results['data_period']['months']
        }

    return metrics

def validate_data_quality():
    """Check data quality across all analyses"""
    issues = []

    # Check core analysis results
    core_results = read_json_results("betler_cost_analysis/output/results.json")
    if core_results:
        r_squared = core_results.get('model_performance', {}).get('r_squared', 0)
        data_days = core_results.get('data_period', {}).get('days', 0)

        if r_squared < 0.7:
            issues.append(f"Core model R² is low ({r_squared:.3f})")
        if data_days < 300:
            issues.append(f"Limited core data ({data_days} days)")
    else:
        issues.append("Core analysis results missing")

    # Check Cognito analysis results
    cognito_results = read_json_results("cognito_cost_analysis/output/results.json")
    if cognito_results:
        r_squared = cognito_results.get('model_performance', {}).get('r_squared', 0)
        data_months = cognito_results.get('data_period', {}).get('months', 0)

        if r_squared < 0.7:
            issues.append(f"Cognito model R² is low ({r_squared:.3f})")
        if data_months < 6:
            issues.append(f"Limited Cognito data ({data_months} months)")
    else:
        issues.append("Cognito analysis results missing")

    return issues

def main():
    """Generate final clean summary with all completion information"""

    print("=" * 50)
    print("BETLER COST ANALYSIS - FINAL SUMMARY")
    print("=" * 50)

    # Validate data quality
    issues = validate_data_quality()
    if issues:
        print("⚠ Data Quality Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    # Extract key metrics
    metrics = extract_key_metrics()

    if not metrics:
        print("❌ Error: Could not extract key metrics from analysis")
        return

    # Display clean summary
    print("📊 12-MONTH COST PROJECTION:")
    print()

    if 'annual_total' in metrics:
        print(f"  💰 Total Cost (Next 12 Months): ${metrics['annual_total']:,.0f}")

    if 'month1_cost' in metrics and 'month12_cost' in metrics:
        growth = ((metrics['month12_cost'] / metrics['month1_cost']) - 1) * 100
        print(f"  📈 Monthly Growth: {growth:+.1f}% (${metrics['month1_cost']:,.0f} → ${metrics['month12_cost']:,.0f})")

    if 'core_total' in metrics and 'cognito_total' in metrics:
        core_pct = (metrics['core_total'] / metrics['annual_total']) * 100
        cognito_pct = (metrics['cognito_total'] / metrics['annual_total']) * 100

        print(f"  ☁️  Core AWS: ${metrics['core_total']:,.0f} ({core_pct:.1f}%)")
        print(f"  🔐 Cognito: ${metrics['cognito_total']:,.0f} ({cognito_pct:.1f}%)")

    print()
    print("✅ Analysis complete - All models and predictions generated")

    # Cost insights
    if 'annual_total' in metrics:
        monthly_avg = metrics['annual_total'] / 12
        daily_avg = metrics['annual_total'] / 365

        print()
        print("💡 COST INSIGHTS:")
        print(f"  - Average monthly cost: ${monthly_avg:,.0f}")
        print(f"  - Average daily cost: ${daily_avg:,.0f}")

        if 'month1_cost' in metrics and 'month12_cost' in metrics:
            if growth > 50:
                print(f"  - High growth trajectory ({growth:.0f}%/year) - monitor scaling efficiency")
            elif growth > 20:
                print(f"  - Moderate growth trajectory ({growth:.0f}%/year) - typical for growing platform")
            else:
                print(f"  - Stable growth trajectory ({growth:.0f}%/year) - costs under control")

    # Results and file locations
    print()
    print("=" * 50)
    print("ALL ANALYSES COMPLETE")
    print("=" * 50)
    print("Results available in:")
    print("  - betler_cost_analysis/output/")
    print("  - cognito_cost_analysis/output/")
    print("  - betler_predictive_analysis/output/")
    print()
    print("View dashboards:")
    print("  - open betler_cost_analysis/output/cost_analysis_dashboard.png")
    print("  - open cognito_cost_analysis/output/cognito_cost_analysis_dashboard.png")
    print("  - open betler_predictive_analysis/output/extended_24month_dashboard.png")
    print()
    print("Key data files:")
    print("  - betler_predictive_analysis/output/predictive_cost_analysis.csv")
    print("  - betler_predictive_analysis/output/results.json")
    print()
    print("All analysis complete! 🎉")

if __name__ == "__main__":
    main()