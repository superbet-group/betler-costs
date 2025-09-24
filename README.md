# Betler Cost Analysis Suite

This repository contains comprehensive cost analysis tools for the Betler platform, including both general AWS costs and specific Amazon Cognito authentication costs.

## Quick Start

Run all analyses with a single command:

```bash
./run_all_analysis.sh
```

This master script will:
1. Start TSH proxy centrally for secure Grafana access
2. Clean up all previous output directories and virtual environments
3. Run the core Betler cost analysis (with proxy skip)
4. Run the Cognito-specific cost analysis (with proxy skip)
5. Generate comprehensive dashboards and predictions
6. Clean up TSH proxy when complete

## Analysis Components

### Core Betler Cost Analysis (`betler_cost_analysis/`)
- Daily AWS cost tracking (excluding tax and Cognito)
- Transaction volume and customer growth correlation
- Business metrics forecasting
- **Output**: `betler_cost_analysis/output/cost_analysis_dashboard.png`

### Cognito Cost Analysis (`cognito_cost_analysis/`)
- Amazon Cognito authentication cost analysis
- Pre/post optimization modeling (July 2025 efficiency improvements)
- Monthly billing cycle alignment with 60% cost reduction prediction
- **Output**: `cognito_cost_analysis/output/cognito_cost_analysis_dashboard.png`

## Individual Analysis

You can also run each analysis separately:

```bash
# Core analysis only
cd betler_cost_analysis
SKIP_TSH_PROXY=true ./core_betler_production_cost_analysis.sh

# Cognito analysis only
cd cognito_cost_analysis
SKIP_TSH_PROXY=true ./cognito_betler_production_cost_analysis.sh
```

## Prerequisites

- `tsh` (Teleport client) for secure Grafana access
- `aws` CLI configured for Cost Explorer access
- `jq` for JSON processing
- Python 3 with `pandas`, `matplotlib`, `numpy`

The scripts automatically set up Python virtual environments and install dependencies.

## Environment Variables

- `SKIP_TSH_PROXY=true` - Skip TSH proxy setup entirely (useful for local development or when proxy is already running)

## Output Files

All analysis results are saved to respective `output/` directories:
- Dashboard visualizations (PNG)
- Raw data (CSV, JSON)
- Statistical analysis (TXT)
- Prediction models and summaries