# Technical Documentation - Cognito Cost Analysis

## Components

**Core Scripts:**
- `cognito_betler_production_cost_analysis.sh` - Main orchestration script with monthly data collection
- `merge_monthly_data.py` - Monthly data consolidation and CSV generation
- `cognito_cost_prediction.py` - Multiple linear regression modeling for monthly data
- `plot_cognito_analysis.py` - 5-panel monthly dashboard visualization (3x2 grid)
- `requirements.txt` - Python dependencies (pandas, matplotlib, numpy)

## Data Sources

**Grafana Metrics (Daily Data Aggregated to Monthly):**
- Daily transaction volume: `sum(increase(wallet[1d]))` - Daily increases summed to monthly totals
- Daily customer volume: `sum(beam_ets{container="cuprer-hot", type="size", table=~"parter_data_[0-9]{3}"})` - Last daily value per month
- **Interval**: 86,400,000ms (1 day), **MaxDataPoints**: 365

**AWS Cost Explorer:**
- Monthly Amazon Cognito costs only for account 493638924148
- Filtered specifically to Cognito service: `{"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Cognito"]}}`
- 12-month historical data with MONTHLY granularity
- Amortized cost metric for consistent billing representation

## Methodology

**Monthly Data Collection:**
1. Queries Grafana for daily platform metrics over the full year
2. Aggregates daily data to calendar months (transactions: sum, customers: last value)
3. Retrieves monthly Cognito costs using Cost Explorer with service-specific filtering
4. Merges data by month (YYYY-MM format) into unified CSV

**Statistical Model:**
- Multiple linear regression: `Cognito_Cost = β₀ + β₁(Monthly_Transactions) + β₂(Monthly_Customers)`
- **Small sample size**: 12 monthly data points (vs 365 daily for core analysis)
- R-squared interpretation adjusted for limited degrees of freedom
- Outlier handling available but likely unnecessary with monthly aggregation

**Visualization:**
- 5-panel monthly dashboard (3x2 grid): transaction volume, customer volume, customer vs cost dual-axis, transaction vs cost dual-axis, Cognito costs with predictions
- Monthly markers and trend lines with dual-axis plots for correlation analysis
- Hybrid prediction model showing pre/post July 2025 optimization

## Key Differences from Daily Analysis

**Temporal Resolution:**
- **Monthly vs Daily**: 12 data points instead of 365
- **Billing Alignment**: Matches Cognito's monthly billing cycle
- **Statistical Power**: Lower due to smaller sample size, but more aligned with cost structure

**Cost Isolation:**
- **Cognito Only**: Excludes all other AWS services (vs daily analysis excluding only tax/Cognito)
- **Authentication Focus**: Direct correlation with user authentication patterns
- **Billing Cycle**: Natural monthly boundaries vs artificial daily aggregation

**Model Expectations:**
- **Lower R-squared**: Expected due to small sample size (12 vs 363 points)
- **Higher Variance**: Monthly aggregation may mask daily fluctuations
- **Business Relevance**: Better aligned with actual Cognito billing patterns

## Environment Requirements

**Dependencies:**
- Same as daily analysis: `tsh`, `aws` CLI, `jq`, Python 3 with scientific stack
- **Query Modification**: Daily data collection with monthly aggregation
- **AWS Filter**: Cognito service only vs exclusion filters

**Output Files:**
- `output/cognito_cost_analysis_dashboard.png` - 5-panel monthly visualization (3x2 grid)
- `output/cognito_cost_analysis.csv` - Monthly dataset
- `output/cognito_regression_results.txt` - Hybrid model statistics with pre/post optimization results
- `output/cognito_summary_statistics.txt` - Monthly summary metrics