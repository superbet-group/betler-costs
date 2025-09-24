# Technical Documentation

## Components

**Core Scripts:**
- `core_betler_production_cost_analysis.sh` - Main orchestration script with TSH proxy management
- `merge_data.py` - Data consolidation and CSV generation from multiple data sources
- `cost_prediction.py` - Multiple linear regression modeling with outlier exclusion
- `plot_analysis.py` - 6-panel dashboard visualization generator
- `requirements.txt` - Python dependencies (pandas, matplotlib, numpy)

## Data Sources

**Grafana Metrics (via TSH proxy):**
- Transaction volume: `sum(increase(wallet[1d]))` over 12 months
- Customer volume: `sum(beam_ets{container="cuprer-hot", type="size", table=~"parter_data_[0-9]{3}"})` over 12 months

**AWS Cost Explorer:**
- Daily amortized costs for account 493638924148
- Filtered to exclude Tax and Amazon Cognito services
- 12-month historical data with daily granularity

## Methodology

**Data Processing:**
1. Queries Grafana for platform metrics using Prometheus API
2. Retrieves AWS costs using Cost Explorer API with service filtering
3. Merges data by date into unified CSV with timestamp alignment
4. Excludes hardcoded outlier dates (billing corrections, data anomalies)

**Statistical Model:**
- Multiple linear regression: `Cost = β₀ + β₁(Transactions) + β₂(Customers)`
- Uses normal equation solver for coefficient calculation
- Excludes outliers: 2025-05-01 (negative cost), 2025-04-17 (customer volume anomaly)
- R-squared: 87.92%, RMSE: 8.14% of mean

**Visualization:**
- 6-panel dashboard: transaction trends, customer growth, daily costs, cost correlations, efficiency metrics, prediction validation
- Actual vs predicted cost overlay for model validation
- Cost per 1000 customers efficiency tracking

## Environment Requirements

**Dependencies:**
- `tsh` (Teleport) for secure Grafana access
- `aws` CLI with betler-prod environment configuration
- `jq` for JSON processing
- Python 3 with pandas, matplotlib, numpy

**Authentication:**
- TSH proxy for Grafana (optional via SKIP_TSH_PROXY=true)
- AWS credentials loaded from `~/.tsh/aws_env_aws-betler-prod`

**Output Files:**
- `output/cost_analysis_dashboard.png` - 6-panel visualization
- `output/cost_analysis.csv` - Merged dataset (365 rows)
- `output/regression_results.txt` - Detailed model statistics
- `output/summary_statistics.txt` - Executive summary metrics