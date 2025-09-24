# Betler Cost Analysis Suite

This repository contains a comprehensive cost analysis and prediction system for the Betler platform, providing historical analysis, future projections, and actionable cost insights.

## 12-Month Cost Forecast

**Total Projected Costs: $8,329,039** (next 12 months)

### Key Projections
- **Customer Growth**: 3% monthly (12.27M → 16.98M customers)
- **Transaction Growth**: 5% monthly (7.27B → projected scaling)
- **Model Accuracy**: R² ~0.88 (88% variance explained), RMSE $872 (8.1%)
- **Based on historic patterns** - does not factor in events such as Romania or other corporate expansions

### Monthly Progression
- **Month 1**: $537,246
- **Month 12**: $870,429
- **Growth trajectory**: 62% increase over 12 months

## Analysis Dashboards

### Core AWS Cost Analysis
![Cost Analysis Dashboard](betler_cost_analysis/output/cost_analysis_dashboard.png)

### Cognito Cost Analysis
![Cognito Cost Dashboard](cognito_cost_analysis/output/cognito_cost_analysis_dashboard.png)

### 24-Month Timeline
![Extended Dashboard](betler_predictive_analysis/output/extended_24month_dashboard.png)

## Methodology

### Core AWS Cost Regression
We run **multiple linear regression** on historical costs for the last 12 months using:
- **Customer volume** - pulled from Grafana metrics
- **Transaction volume** - pulled from Grafana metrics
- **Historical AWS costs** - from AWS Cost Explorer API (excluding tax and Cognito)

The regression model provides the data accuracy (R² score) and generates predictions based on volume correlations.

### Cognito Cost Analysis
We perform a **similar linear regression for Cognito costs**, with key differences:
- **Monthly aggregation** - Cognito has large monthly billing cycles rather than daily
- **Hybrid optimization model** - In July 2025, major optimization occurred (60% cost reduction)
- **Post-optimization adjustment** - Regression predictions are multiplied by optimization factor (0.4x)

**Note**: We are unsure if the Cognito predictions will remain accurate going forward, but it represents a small proportion of overall costs.

## Quick Start

Run the complete analysis pipeline with a single command:

```bash
./run_all_analysis.sh
```

This master script will:
1. Start TSH proxy centrally for secure Grafana access
2. Clean up all previous output directories and virtual environments
3. Run the core Betler cost analysis with regression modeling
4. Run the Cognito-specific cost analysis with hybrid optimization modeling
5. Generate predictive cost projections with configurable growth parameters
6. Create comprehensive dashboards including 24-month timeline visualization
7. Display 12-month forward-looking cost estimates
8. Clean up TSH proxy when complete

## Analysis Components

### 1. Core AWS Cost Analysis (`betler_cost_analysis/`)
- Daily AWS cost extraction (excluding tax and Cognito services)
- Multiple linear regression: `Cost = α + β₁×Transactions + β₂×Customers`
- Model validation with R² accuracy reporting
- **Outputs**: Dashboard, regression model, historical data CSV, results JSON

### 2. Cognito Cost Analysis (`cognito_cost_analysis/`)
- Amazon Cognito authentication cost analysis with monthly aggregation
- Hybrid modeling: pre-July 2025 regression × post-optimization factor (0.4)
- Separate validation for optimization effectiveness detection
- **Outputs**: Monthly dashboard, hybrid regression model, cost projections, results JSON

### 3. Predictive Cost Analysis (`betler_predictive_analysis/`)
- Forward-looking cost projections using regression models from both analyses
- Configurable growth parameters (customer growth, transaction growth, projection period)
- Combined cost modeling: Core AWS + Cognito with optimization factors
- **Outputs**: Monthly projection CSV, structured results JSON

### 4. Extended 24-Month Dashboard
- Unified visualization showing 12 months historical + 12 months predictive data
- Triple-axis plot combining transaction volume, customer volume, and AWS costs
- Clear delineation between actual and predicted data
- **Output**: Single comprehensive 24-month timeline visualization

## Individual Analysis

You can also run each analysis separately:

```bash
# Core analysis only
cd betler_cost_analysis
SKIP_TSH_PROXY=true ./core_betler_production_cost_analysis.sh

# Cognito analysis only
cd cognito_cost_analysis
SKIP_TSH_PROXY=true ./cognito_betler_production_cost_analysis.sh

# Predictive analysis only (requires core and Cognito to be run first)
cd betler_predictive_analysis
./run_predictive_analysis.sh

# Custom predictive analysis with growth parameters
./run_predictive_analysis.sh --customer-growth 0.04 --transaction-growth 0.06 --months 18
```

## Prerequisites

- `tsh` (Teleport client) for secure Grafana access
- `aws` CLI configured for Cost Explorer access
- `jq` for JSON processing
- Python 3 with `pandas`, `matplotlib`, `numpy`

The scripts automatically set up Python virtual environments and install dependencies.

## Environment Variables

- `SKIP_TSH_PROXY=true` - Skip TSH proxy setup entirely (useful for local development or when proxy is already running)

## Model Accuracy

Both regression models include accuracy metrics:
- **R² (coefficient of determination)** - proportion of variance explained by the model
- **RMSE (root mean square error)** - prediction error magnitude
- **Data quality warnings** - alerts for low R² or insufficient data

Typical performance:
- Core AWS model: R² ~0.88 (88% variance explained)
- Cognito model: R² ~0.87 (87% variance explained, pre-optimization data)

## Output Files

All analysis results are saved to respective `output/` directories:

### Core Analysis (`betler_cost_analysis/output/`)
- `cost_analysis_dashboard.png` - 6-panel historical dashboard
- `cost_analysis.csv` - Daily data with transactions, customers, costs
- `core_regression_model.json` - Regression coefficients for predictions
- `results.json` - Structured key metrics and model performance

### Cognito Analysis (`cognito_cost_analysis/output/`)
- `cognito_cost_analysis_dashboard.png` - Monthly Cognito cost trends
- `cognito_cost_analysis.csv` - Monthly aggregated data
- `cognito_regression_model.json` - Hybrid model with optimization factors
- `results.json` - Structured metrics including optimization detection

### Predictive Analysis (`betler_predictive_analysis/output/`)
- `extended_24month_dashboard.png` - Unified 24-month timeline visualization
- `predictive_cost_analysis.csv` - Monthly forward projections
- `results.json` - Structured 12-month projections and growth parameters

## 12-Month Cost Estimates

The master script automatically displays forward-looking cost estimates:
- **Total 12-month projection**
- **Monthly cost progression** (Month 1 vs Month 12)
- **Service breakdown** (Core AWS vs Cognito costs)
- **Growth impact analysis**

Example output:
```
======================================
12-MONTH COST ESTIMATE
======================================
📊 FORWARD-LOOKING COST PROJECTION:

  💰 Next 12 Months Total: $8,970,674.79
  📈 Month 1 Cost: $537,246.26
  📈 Month 12 Cost: $991,341.94
  🔐 Cognito Costs (12 months): $352,206.64
  ☁️  Core AWS Costs (12 months): $8,618,468.15
```