# Betler Production Cost Analysis

This analysis correlates AWS operational costs with platform usage metrics to provide accurate cost forecasting for the Betler production environment.

## Summary

**Financial Overview:**
- **$3.89M annual operational AWS costs** (excluding tax/Cognito)
- **$321K/month** current projection
- **87.92% predictive accuracy** for cost forecasting

**Growth Metrics:**
- **343% transaction volume growth** over 12 months
- **78% customer base growth** from 7M to 12.6M users
- Cost drivers: $0.0016 per customer unit, $0.000019 per transaction

**Scope:**
This analysis focuses on core operational AWS costs and excludes tax charges, Cognito authentication costs, development environments, and billing anomalies to provide clean predictive capabilities for production capacity planning.

## Usage

```bash
SKIP_TSH_PROXY=true ./core_betler_production_cost_analysis.sh
```

See `TECHNICAL.md` for detailed information about the scripts and methodology.