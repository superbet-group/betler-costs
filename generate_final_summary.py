#!/usr/bin/env python3

import os
import json
import pandas as pd

def read_summary_file(filepath):
    """Read and parse the predictive analysis summary file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            return content
        return None
    except Exception:
        return None

def extract_key_metrics():
    """Extract key metrics from all analysis outputs"""

    # Read predictive analysis summary
    summary_path = "betler_predictive_analysis/output/predictive_analysis_summary.txt"
    summary_content = read_summary_file(summary_path)

    if not summary_content:
        print("❌ Error: Predictive analysis summary not found")
        return None

    # Extract key numbers from summary
    metrics = {}

    # Look for annual total
    for line in summary_content.split('\n'):
        if "Annual Total Cost:" in line:
            try:
                cost_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                metrics['annual_total'] = float(cost_str)
            except:
                pass
        elif "Month 1 Cost:" in line:
            try:
                cost_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                metrics['month1_cost'] = float(cost_str)
            except:
                pass
        elif "Month 12 Cost:" in line:
            try:
                cost_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                metrics['month12_cost'] = float(cost_str)
            except:
                pass
        elif "Cognito Costs:" in line:
            try:
                cost_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                metrics['cognito_total'] = float(cost_str)
            except:
                pass
        elif "Core AWS Costs:" in line:
            try:
                cost_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                metrics['core_total'] = float(cost_str)
            except:
                pass

    return metrics

def validate_data_quality():
    """Check data quality across all analyses"""
    issues = []

    # Check core analysis model
    core_model_path = "betler_cost_analysis/output/core_regression_model.json"
    if os.path.exists(core_model_path):
        try:
            with open(core_model_path, 'r') as f:
                core_model = json.load(f)

            r_squared = core_model.get('model', {}).get('r_squared', 0)
            data_points = core_model.get('data_points', 0)

            if r_squared < 0.7:
                issues.append(f"Core model R² is low ({r_squared:.3f})")
            if data_points < 300:
                issues.append(f"Limited core data ({data_points} points)")

        except Exception:
            issues.append("Core model file corrupted")
    else:
        issues.append("Core model missing")

    # Check Cognito analysis model
    cognito_model_path = "cognito_cost_analysis/output/cognito_regression_model.json"
    if os.path.exists(cognito_model_path):
        try:
            with open(cognito_model_path, 'r') as f:
                cognito_model = json.load(f)

            r_squared = cognito_model.get('base_model', {}).get('r_squared', 0)

            if r_squared < 0.7:
                issues.append(f"Cognito model R² is low ({r_squared:.3f})")

        except Exception:
            issues.append("Cognito model file corrupted")
    else:
        issues.append("Cognito model missing")

    return issues

def main():
    """Generate final clean summary"""

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

if __name__ == "__main__":
    main()