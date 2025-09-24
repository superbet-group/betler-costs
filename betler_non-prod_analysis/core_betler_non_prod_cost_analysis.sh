#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Non-prod cost analysis script - simplified environment-based cost analysis
# Usage: ./core_betler_non_prod_cost_analysis.sh

# Check if we should skip the proxy (default: use proxy)
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    # Start AWS TSH proxy for dev environment
    AWS_RESULT=$(../start_tsh_aws_proxy.sh aws-betler-dev)
    AWS_TSH_PID=$(echo $AWS_RESULT | awk '{print $1}')
    AWS_ENV_FILE=$(echo $AWS_RESULT | awk '{print $2}')

    # Wait briefly and check if AWS env file was created
    sleep 2
    if [ ! -f "$AWS_ENV_FILE" ]; then
        echo "Error: AWS TSH proxy failed for dev - env file not created at $AWS_ENV_FILE" >&2
        exit 1
    fi
fi

# Create output directory if it doesn't exist
mkdir -p output

# Set up Python virtual environment if it doesn't exist
if [ ! -d "../venv" ]; then
    echo "Setting up Python virtual environment..."
    cd ..
    python3 -m venv venv
    source venv/bin/activate
    pip install -r betler_non-prod_analysis/requirements.txt > /dev/null 2>&1
    deactivate
    cd betler_non-prod_analysis
    echo "✓ Virtual environment ready"
fi

# Get AWS costs for non-prod environments grouped by environment tag
echo "Fetching AWS costs for non-prod environments grouped by environment tag..."

# Load AWS environment variables for staging/dev
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    export $(cat "$AWS_ENV_FILE" | tr -d \")
else
    export $(cat ~/.tsh/aws_env_aws-betler-dev | tr -d \")
fi

# Get costs grouped by environment tag with daily granularity
aws ce get-cost-and-usage \
    --time-period Start=$(date -d "12 months ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity DAILY \
    --metrics AmortizedCost \
    --group-by Type=TAG,Key=environment \
    --filter '{
        "Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Tax"]}}
    }' > output/aws_costs_by_environment.json

# Also get overall daily costs without grouping for trend analysis
aws ce get-cost-and-usage \
    --time-period Start=$(date -d "12 months ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity DAILY \
    --metrics AmortizedCost \
    --filter '{
        "Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Tax"]}}
    }' > output/aws_daily_costs.json

# Calculate total cost from daily data
TOTAL_COST=$(jq -r '[.ResultsByTime[].Total.AmortizedCost.Amount | tonumber] | add' output/aws_daily_costs.json)
echo "Total AWS cost for last 12 months (excluding tax): \$$(printf "%.2f" $TOTAL_COST)"

# Show environment breakdown
echo ""
echo "Cost breakdown by environment:"
jq -r '
.ResultsByTime[] |
.TimePeriod.Start as $date |
.Groups[] |
select(.Keys[0] != null and .Keys[0] != "") |
"\($date): \(.Keys[0]) = $\(.Metrics.AmortizedCost.Amount)"
' output/aws_costs_by_environment.json | tail -10

# Run simplified non-prod analysis
echo ""
echo "Running non-prod cost analysis..."
../venv/bin/python non_prod_analysis.py

# Terminate the background process if it was started
if [ "${SKIP_TSH_PROXY:-false}" != "true" ]; then
    if [ ! -z "${AWS_TSH_PID:-}" ]; then
        kill $AWS_TSH_PID 2>/dev/null || true
    fi
fi