#!/bin/bash

# Predictive Cost Analysis Script
# Usage: ./run_predictive_analysis.sh [--customer-growth 0.03] [--transaction-growth 0.05] [--months 12]

# Enable strict error handling
set -euo pipefail

echo "=========================================="
echo "BETLER PREDICTIVE COST ANALYSIS"
echo "=========================================="

# Check if analysis outputs exist
if [ ! -f "../betler_cost_analysis/output/core_regression_model.json" ]; then
    echo "Error: Core analysis model not found."
    echo "Please run the core analysis first: cd ../betler_cost_analysis && ./core_betler_production_cost_analysis.sh"
    exit 1
fi

if [ ! -f "../cognito_cost_analysis/output/cognito_regression_model.json" ]; then
    echo "Error: Cognito analysis model not found."
    echo "Please run the Cognito analysis first: cd ../cognito_cost_analysis && ./cognito_betler_production_cost_analysis.sh"
    exit 1
fi

# Set up virtual environment for Python dependencies
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing requirements..."
    venv/bin/pip install -r requirements.txt
fi

# Run predictive analysis with all provided arguments
echo "Running predictive cost analysis..."
venv/bin/python predictive_cost_analysis.py "$@"

echo ""
echo "=========================================="
echo "PREDICTIVE ANALYSIS COMPLETE"
echo "=========================================="
echo "Files created:"
echo "  - output/predictive_cost_dashboard.png"
echo "  - output/predictive_cost_analysis.csv"
echo "  - output/predictive_analysis_summary.txt"
echo ""
echo "View dashboard:"
echo "  open output/predictive_cost_dashboard.png"