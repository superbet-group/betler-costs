#!/bin/bash

# Master script to run all cost analysis
# This script manages TSH proxy centrally and runs both core and Cognito analysis

# Enable strict error handling
set -euo pipefail


# Start TSH proxies centrally (unless explicitly skipped)
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    GRAFANA_TSH_PID=$(./start_tsh_grafana_proxy.sh)
    AWS_RESULT=$(./start_tsh_aws_proxy.sh aws-betler-prod)
    AWS_TSH_PID=$(echo $AWS_RESULT | awk '{print $1}')
    AWS_ENV_FILE=$(echo $AWS_RESULT | awk '{print $2}')

    # Wait briefly and check if AWS env file was created
    sleep 2
    if [ ! -f "$AWS_ENV_FILE" ]; then
        echo "Error: AWS TSH proxy failed - env file not created at $AWS_ENV_FILE" >&2
        exit 1
    fi
    echo ""
fi

# Clean up all output directories and virtual environment
echo "Cleaning up previous analysis results..."
rm -rf betler_cost_analysis/output cognito_cost_analysis/output betler_predictive_analysis/output
rm -rf venv
echo "Cleanup complete."
echo ""

# Set up shared Python virtual environment
echo "Setting up shared Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1
deactivate
echo "Shared virtual environment ready."
echo ""

# Run core Betler cost analysis with proxy skip
echo "=========================================="
echo "RUNNING CORE BETLER COST ANALYSIS"
echo "=========================================="
cd betler_cost_analysis
SKIP_TSH_PROXY=true ./core_betler_production_cost_analysis.sh
cd ..
echo "Core analysis complete."
echo ""

# Run Cognito cost analysis with proxy skip
echo "=========================================="
echo "RUNNING COGNITO COST ANALYSIS"
echo "=========================================="
cd cognito_cost_analysis
SKIP_TSH_PROXY=true ./cognito_betler_production_cost_analysis.sh
cd ..
echo "Cognito analysis complete."
echo ""

# Run predictive analysis
echo "=========================================="
echo "RUNNING PREDICTIVE COST ANALYSIS"
echo "=========================================="
cd betler_predictive_analysis

# Run predictive analysis with default parameters using shared venv
echo "Running predictive cost analysis..."
../venv/bin/python predictive_cost_analysis.py

echo "Running extended 24-month dashboard..."
../venv/bin/python extended_dashboard.py

echo "✓ Predictive cost analysis complete"
cd ..

# Generate final clean summary
echo "==========================================="
echo "GENERATING FINAL SUMMARY"
echo "==========================================="
./venv/bin/python generate_final_summary.py
echo ""

# Clean up TSH proxies (only if we started them)
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    if [ ! -z "${GRAFANA_TSH_PID:-}" ]; then
        kill $GRAFANA_TSH_PID 2>/dev/null || true
    fi
    if [ ! -z "${AWS_TSH_PID:-}" ]; then
        kill $AWS_TSH_PID 2>/dev/null || true
    fi
fi

echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================="
echo "Results available in:"
echo "  - betler_cost_analysis/output/"
echo "  - cognito_cost_analysis/output/"
echo "  - betler_predictive_analysis/output/"
echo ""
echo "View dashboards:"
echo "  - open betler_cost_analysis/output/cost_analysis_dashboard.png"
echo "  - open cognito_cost_analysis/output/cognito_cost_analysis_dashboard.png"
echo "  - open betler_predictive_analysis/output/predictive_cost_dashboard.png"
echo "  - open betler_predictive_analysis/output/extended_24month_dashboard.png"
echo ""

# Extract and display 12-month cost estimate
echo "======================================"
echo "12-MONTH COST ESTIMATE"
echo "======================================"
if [ -f "betler_predictive_analysis/output/predictive_analysis_summary.txt" ]; then
    echo "📊 FORWARD-LOOKING COST PROJECTION:"
    echo ""
    # Extract key metrics from the summary file
    ANNUAL_COST=$(grep "Annual Total Cost:" betler_predictive_analysis/output/predictive_analysis_summary.txt | awk '{print $4}')
    MONTH1_COST=$(grep "Total Cost (Month 1):" betler_predictive_analysis/output/predictive_analysis_summary.txt | awk '{print $5}')
    MONTH12_COST=$(grep "Total Cost (Month 12):" betler_predictive_analysis/output/predictive_analysis_summary.txt | awk '{print $5}')
    COGNITO_TOTAL=$(grep "Cognito Costs:" betler_predictive_analysis/output/predictive_analysis_summary.txt | awk '{print $3}')
    CORE_TOTAL=$(grep "Core AWS Costs:" betler_predictive_analysis/output/predictive_analysis_summary.txt | awk '{print $4}')

    echo "  💰 Next 12 Months Total: $ANNUAL_COST"
    echo "  📈 Month 1 Cost: $MONTH1_COST"
    echo "  📈 Month 12 Cost: $MONTH12_COST"
    echo "  🔐 Cognito Costs (12 months): $COGNITO_TOTAL"
    echo "  ☁️  Core AWS Costs (12 months): $CORE_TOTAL"
    echo ""
    echo "  📊 View detailed breakdown: betler_predictive_analysis/output/predictive_analysis_summary.txt"
    echo "  📈 Monthly data: betler_predictive_analysis/output/predictive_cost_analysis.csv"
    echo "  📊 24-month visualization: betler_predictive_analysis/output/extended_24month_dashboard.png"
else
    echo "⚠ Cost estimate not available - predictive analysis may have failed"
fi
echo ""
echo "All analysis complete! 🎉"
echo "Check the output directories for detailed results and visualizations."