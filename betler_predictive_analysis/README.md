# Betler Predictive Cost Analysis

Forward-looking cost projections based on business growth parameters and existing regression models.

## Overview

This predictive analysis system combines regression models from both core AWS cost analysis and Cognito cost analysis to generate accurate cost projections based on configurable growth parameters.

## Prerequisites

**Required Analysis Data:**
- Core AWS cost analysis must be run first (generates `core_regression_model.json`)
- Cognito cost analysis must be run first (generates `cognito_regression_model.json`)

## Usage

### Basic Usage (Default Parameters)
```bash
./run_predictive_analysis.sh
```

**Default Growth Rates:**
- Customer Growth: 3% per month
- Transaction Growth: 5% per month
- Projection Period: 12 months

### Custom Growth Parameters
```bash
./run_predictive_analysis.sh --customer-growth 0.04 --transaction-growth 0.06 --months 18
```

### Available Parameters
- `--customer-growth`: Monthly customer growth rate (e.g., 0.03 = 3%)
- `--transaction-growth`: Monthly transaction growth rate (e.g., 0.05 = 5%)
- `--months`: Number of months to project (default: 12)

## How It Works

### 1. Model Loading
- Loads regression models from JSON files created by core and Cognito analysis
- Uses exact model coefficients and parameters from historical data analysis

### 2. Growth Projection
- Applies compound growth rates to current baseline metrics
- Projects both customer count and transaction volume month-by-month

### 3. Cost Prediction
- **Cognito Costs**: Uses hybrid model with post-July 2025 optimization (60% reduction)
- **Core AWS Costs**: Uses daily cost model scaled to monthly projections
- **Combined**: Total predicted platform costs

### 4. Visualization
Generates comprehensive 6-panel dashboard:
1. Projected Customer Growth
2. Projected Transaction Growth
3. Cost Breakdown Over Time (Cognito vs Core AWS vs Total)
4. Month-over-Month Cost Growth Rate
5. Cost per Customer Trend
6. 12-Month Cost Projection Summary

## Output Files

All files are saved to `output/` directory:

- **`predictive_cost_dashboard.png`** - 6-panel visualization dashboard
- **`predictive_cost_analysis.csv`** - Detailed monthly projections data
- **`predictive_analysis_summary.txt`** - Key metrics and annual projections

## Example Scenarios

### Conservative Growth
```bash
./run_predictive_analysis.sh --customer-growth 0.02 --transaction-growth 0.03
```

### Aggressive Growth
```bash
./run_predictive_analysis.sh --customer-growth 0.06 --transaction-growth 0.08
```

### Long-term Planning
```bash
./run_predictive_analysis.sh --months 24 --customer-growth 0.035
```

## Model Accuracy

The predictions are built on regression models with:
- **Core AWS Model**: Based on daily cost data with R² values from historical analysis
- **Cognito Model**: Hybrid approach accounting for July 2025 optimization (86.68% R² pre-optimization)
- **Growth Assumptions**: Linear compound growth applied to baseline metrics

Prediction accuracy depends on:
1. Stability of underlying cost relationships
2. Accuracy of provided growth parameters
3. Continued effectiveness of optimization strategies