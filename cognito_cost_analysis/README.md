# Betler Cognito Cost Analysis

This analysis examines monthly Amazon Cognito authentication costs in relation to platform usage metrics to understand billing patterns and predict authentication service expenses.

## Summary

**Cognito Billing Pattern:**
- **Monthly billing cycles** - Cognito charges appear on the 1st of each month
- **12 data points** - Monthly aggregated analysis over the past year
- Authentication cost correlation with user activity

**Analysis Scope:**
- Daily transaction and customer data aggregated to calendar months
- Amazon Cognito service costs only (isolated from other AWS services)
- Pre and post-optimization periods (July 2025 optimization)

**Key Findings:**
- Pre-optimization: 86.68% R-squared prediction accuracy
- July 2025 optimization reduced costs by 60%
- Customer volume is primary cost driver (not transaction frequency)

## Prediction Model

**Simple Hybrid Approach:**
The model uses a straightforward approach that accounts for the July 2025 authentication optimization.

**Model Logic:**
1. **Pre-July 2025**: Use base regression model trained on stable historical data
2. **July 2025+**: Multiply base predictions by 0.4 (60% cost reduction)

**Base Model:**
```
Cognito_Cost = Intercept + (Coefficient_1 × Transactions) + (Coefficient_2 × Customers)
```

**Future Predictions:**
```
Predicted_Cost = Base_Model_Cost × 0.4
```

**Why This Works:**
- **Predictable Inputs**: Customer and transaction volumes can be forecasted months ahead
- **Step-Function Change**: July optimization created a clear before/after cost structure
- **Business-Focused**: Model scales with business growth using optimized cost structure

## Usage

```bash
SKIP_TSH_PROXY=true ./cognito_betler_production_cost_analysis.sh
```

See `TECHNICAL.md` for detailed information about the monthly aggregation methodology and scripts.