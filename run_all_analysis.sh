#!/bin/bash

# Master script to run all cost analysis
# This script manages TSH proxy centrally and runs both core and Cognito analysis

# Enable strict error handling
set -euo pipefail

echo "=========================================="
echo "BETLER COST ANALYSIS - MASTER SCRIPT"
echo "=========================================="
echo "This script will:"
echo "1. Start TSH proxy for Grafana access"
echo "2. Clean up all previous output directories"
echo "3. Run core Betler cost analysis"
echo "4. Run Cognito cost analysis"
echo "5. Clean up TSH proxy"
echo ""

# Start TSH proxy centrally (unless explicitly skipped)
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    echo "Starting TSH proxy for Grafana access..."
    tsh proxy app grafana --port=8080 &
    TSH_PID=$!
    echo "TSH proxy started (PID: $TSH_PID)"
    echo ""
fi

# Clean up all output directories
echo "Cleaning up previous analysis results..."
if [ -d "betler_cost_analysis/output" ]; then
    echo "  - Removing betler_cost_analysis/output/"
    rm -rf betler_cost_analysis/output
fi

if [ -d "cognito_cost_analysis/output" ]; then
    echo "  - Removing cognito_cost_analysis/output/"
    rm -rf cognito_cost_analysis/output
fi

if [ -d "betler_cost_analysis/venv" ]; then
    echo "  - Removing betler_cost_analysis/venv/"
    rm -rf betler_cost_analysis/venv
fi

if [ -d "cognito_cost_analysis/venv" ]; then
    echo "  - Removing cognito_cost_analysis/venv/"
    rm -rf cognito_cost_analysis/venv
fi

echo "Cleanup complete."
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

# Clean up previous predictive results
if [ -d "output" ]; then
    echo "  - Removing betler_predictive_analysis/output/"
    rm -rf output
fi

if [ -d "venv" ]; then
    echo "  - Removing betler_predictive_analysis/venv/"
    rm -rf venv
fi

# Run predictive analysis with default parameters
echo "Running predictive cost analysis..."
python predictive_cost_analysis.py

echo "Running extended 24-month dashboard..."
python extended_dashboard.py

echo "✓ Predictive cost analysis complete"
cd ..
echo ""

# Clean up TSH proxy
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    echo "Stopping TSH proxy..."
    kill $TSH_PID 2>/dev/null || true
    echo "TSH proxy stopped."
    echo ""
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