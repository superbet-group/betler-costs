#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Cognito cost analysis script - Monthly data analysis
# Usage: ./cognito_betler_production_cost_analysis.sh

query_grafana() {
    local query="$1"
    local start_time="$2"
    local end_time="$3"

    # Create JSON payload with daily intervals (same as main analysis)
    local json_payload=$(jq -n \
      --arg query "$query" \
      --arg start_time "$start_time" \
      --arg end_time "$end_time" \
      '{
        "queries": [{
          "refId": "A",
          "expr": $query,
          "range": true,
          "instant": false,
          "datasource": {
            "type": "prometheus",
            "uid": "betler-production"
          },
          "intervalMs": 86400000,
          "maxDataPoints": 365
        }],
        "from": $start_time,
        "to": $end_time
      }')

    curl -s "http://localhost:8080/api/ds/query?ds_type=prometheus" \
      -X POST \
      -H "Content-Type: application/json" \
      -H "x-datasource-uid: betler-production" \
      --data-raw "$json_payload"
}

# Check if we should skip the proxy (default: use proxy)
if [ "${SKIP_TSH_PROXY}" != "true" ]; then
    # Start tsh proxy in background and capture PID
    tsh proxy app grafana --port=8080 &
    TSH_PID=$!
fi

# Calculate timestamps for last 12 months, starting at midnight yesterday (same as main analysis)
END_TIME=$(date -d "today 00:00" +%s%3N)
START_TIME=$(date -d "12 months ago today 00:00" +%s%3N)

# Create output directory if it doesn't exist
mkdir -p output

# Query daily wallet data (to be aggregated monthly later)
query_grafana "sum(increase(wallet[1d]))" "$START_TIME" "$END_TIME" > output/daily_transaction_volume_query.json

# Query daily beam_ets data (to be aggregated monthly later)
query_grafana 'sum(beam_ets{container="cuprer-hot", type="size", table=~"parter_data_[0-9]{3}"})' "$START_TIME" "$END_TIME" > output/daily_customer_volume_query.json

# Extract and display most recent values for debugging
LATEST_WALLET=$(jq -r '.results.A.frames[0].data.values[1][-1] // "No data"' output/daily_transaction_volume_query.json)
LATEST_CUSTOMER=$(jq -r '.results.A.frames[0].data.values[1][-1] // "No data"' output/daily_customer_volume_query.json)
LATEST_DATE=$(jq -r '.results.A.frames[0].data.values[0][-1] // "No data"' output/daily_transaction_volume_query.json)

# Convert timestamp to readable date
if [ "$LATEST_DATE" != "No data" ]; then
    READABLE_DATE=$(date -d "@$(echo $LATEST_DATE | cut -c1-10)" '+%Y-%m')
else
    READABLE_DATE="No data"
fi

echo "Latest date: $READABLE_DATE"
echo "Latest daily transaction volume: $LATEST_WALLET"
echo "Latest daily customer volume: $LATEST_CUSTOMER"

# Get AWS Cognito costs only for the last 12 months
echo
echo "Fetching Amazon Cognito costs for account 493638924148..."

# Load AWS environment variables and get Cognito cost data per month
export $(cat ~/.tsh/aws_env_aws-betler-prod | tr -d \")

aws ce get-cost-and-usage \
    --time-period Start=$(date -d "12 months ago" +%Y-%m-01),End=$(date +%Y-%m-01) \
    --granularity MONTHLY \
    --metrics AmortizedCost \
    --filter '{"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Cognito"]}}' > output/aws_monthly_cognito_costs.json

# Calculate total Cognito cost from monthly data
TOTAL_COGNITO_COST=$(jq -r '[.ResultsByTime[].Total.AmortizedCost.Amount | tonumber] | add' output/aws_monthly_cognito_costs.json)
echo "Total AWS Cognito cost for last 12 months: \$$(printf "%.2f" $TOTAL_COGNITO_COST)"

# Merge all monthly data into a single CSV file using Python
echo "Merging monthly data into CSV..."
venv/bin/python merge_monthly_data.py

# Run Cognito cost prediction analysis
echo "Running Cognito cost prediction analysis..."
venv/bin/python cognito_cost_prediction.py

# Create Cognito visualization dashboard
echo "Creating Cognito visualization dashboard..."
venv/bin/python plot_cognito_analysis.py

# Terminate the background process if it was started
if [ "${SKIP_TSH_PROXY}" != "true" ]; then
    kill $TSH_PID 2>/dev/null
fi