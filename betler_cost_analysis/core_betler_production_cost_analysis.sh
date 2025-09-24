#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Simple cost analysis script
# Usage: ./cost_analysis.sh

query_grafana() {
    local query="$1"
    local start_time="$2"
    local end_time="$3"

    # Create JSON payload with proper escaping
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

# Calculate timestamps for last 12 months, starting at midnight yesterday
END_TIME=$(date -d "today 00:00" +%s%3N)
START_TIME=$(date -d "12 months ago today 00:00" +%s%3N)

# Create output directory if it doesn't exist
mkdir -p output

# Query wallet data and save results directly to file
query_grafana "sum(increase(wallet[1d]))" "$START_TIME" "$END_TIME" > output/transaction_volume_query.json

# Query beam_ets data and save results directly to file
query_grafana 'sum(beam_ets{container="cuprer-hot", type="size", table=~"parter_data_[0-9]{3}"})' "$START_TIME" "$END_TIME" > output/customer_volume_query.json

# Extract and display most recent values for debugging
LATEST_WALLET=$(jq -r '.results.A.frames[0].data.values[1][-1] // "No data"' output/transaction_volume_query.json)
LATEST_CUSTOMER=$(jq -r '.results.A.frames[0].data.values[1][-1] // "No data"' output/customer_volume_query.json)
LATEST_DATE=$(jq -r '.results.A.frames[0].data.values[0][-1] // "No data"' output/transaction_volume_query.json)

# Convert timestamp to readable date
if [ "$LATEST_DATE" != "No data" ]; then
    READABLE_DATE=$(date -d "@$(echo $LATEST_DATE | cut -c1-10)" '+%Y-%m-%d')
else
    READABLE_DATE="No data"
fi

echo "Latest date: $READABLE_DATE"
echo "Latest transaction volume: $LATEST_WALLET"
echo "Latest customer volume: $LATEST_CUSTOMER"

# Get AWS costs for betler platform
echo
echo "Fetching AWS costs for account 493638924148..."

# Load AWS environment variables and get cost data per day for the last 12 months
export $(cat ~/.tsh/aws_env_aws-betler-prod | tr -d \")

aws ce get-cost-and-usage \
    --time-period Start=$(date -d "12 months ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity DAILY \
    --metrics AmortizedCost \
    --filter '{
        "And": [
            {"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Tax"]}}},
            {"Not": {"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Cognito"]}}}
        ]
    }' > output/aws_daily_costs.json

# Calculate total cost from daily data
TOTAL_COST=$(jq -r '[.ResultsByTime[].Total.AmortizedCost.Amount | tonumber] | add' output/aws_daily_costs.json)
echo "Total AWS cost for last 12 months (excluding tax and Cognito): \$$(printf "%.2f" $TOTAL_COST)"

# Merge all data into a single CSV file using Python
echo "Merging data into CSV..."
venv/bin/python merge_data.py

# Run cost prediction analysis
echo "Running cost prediction analysis..."
venv/bin/python cost_prediction.py

# Create visualization dashboard
echo "Creating visualization dashboard..."
venv/bin/python plot_analysis.py

# Terminate the background process if it was started
if [ "${SKIP_TSH_PROXY}" != "true" ]; then
    kill $TSH_PID 2>/dev/null
fi