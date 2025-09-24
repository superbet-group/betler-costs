# Technical Implementation Details

## Methodology

### Core AWS Cost Regression
We run **multiple linear regression** on historical costs for the last 12 months using:
- **Customer volume** - pulled from Grafana metrics
- **Transaction volume** - pulled from Grafana metrics
- **Historical AWS costs** - from AWS Cost Explorer API (excluding tax and Cognito)

The regression model provides the data accuracy (R² score) and generates predictions based on volume correlations.

**Mathematical Model:**
```
Cost = α + β₁×Transactions + β₂×Customers
```

### Cognito Cost Analysis
We perform a **similar linear regression for Cognito costs**, with key differences:
- **Monthly aggregation** - Cognito has large monthly billing cycles rather than daily
- **Hybrid optimization model** - In July 2025, major optimization occurred (60% cost reduction)
- **Post-optimization adjustment** - Regression predictions are multiplied by optimization factor (0.4x)

**Note**: We are unsure if the Cognito predictions will remain accurate going forward, but it represents a small proportion of overall costs.

## Model Accuracy

Both regression models include accuracy metrics:
- **R² (coefficient of determination)** - proportion of variance explained by the model
- **RMSE (root mean square error)** - prediction error magnitude
- **Data quality warnings** - alerts for low R² or insufficient data

Typical performance:
- Core AWS model: R² ~0.88 (88% variance explained)
- Cognito model: R² ~0.87 (87% variance explained, pre-optimization data)

## Data Sources

### Grafana Metrics
- **Transaction Volume**: `sum(increase(wallet[1d]))` - Daily transaction counts
- **Customer Volume**: `sum(beam_ets{container="cuprer-hot", type="size", table=~"parter_data_[0-9]{3}"})` - Active customer base size
- **Access**: Via TSH proxy to `betler-production` datasource

### AWS Cost Explorer
- **Core AWS Costs**: Daily granularity, excluding tax and Cognito services
- **Cognito Costs**: Monthly granularity, Amazon Cognito service only
- **Access**: Via AWS CLI with Cost Explorer API permissions

## Technical Architecture

### Analysis Components

#### 1. Core AWS Cost Analysis (`betler_cost_analysis/`)
- **Data Processing**: Daily cost extraction with outlier removal
- **Regression**: Multiple linear regression using normal equation method
- **Model Export**: JSON format for predictive analysis consumption
- **Validation**: R² threshold warnings, data completeness checks

#### 2. Cognito Cost Analysis (`cognito_cost_analysis/`)
- **Data Aggregation**: Daily metrics aggregated to monthly
- **Hybrid Modeling**:
  - Train on pre-July 2025 data
  - Apply 0.4 multiplier for post-optimization predictions
- **Optimization Detection**: Automatic detection of cost reduction effectiveness

#### 3. Predictive Cost Analysis (`betler_predictive_analysis/`)
- **Model Loading**: Imports regression models from JSON
- **Growth Parameters**: Configurable monthly growth rates
- **Projection**: 12-month forward predictions with compound growth
- **Output**: Structured JSON results for consumption

## Data Flow

```
Grafana (TSH) → Daily Metrics → CSV → Regression Analysis → JSON Model
                                                              ↓
AWS Cost Explorer → Historical Costs → CSV → Model Training → Predictions
```

## File Structure

### Input Data
- `cost_analysis.csv` - Daily core AWS data
- `cognito_cost_analysis.csv` - Monthly Cognito data
- TSH proxy authentication files

### Model Files
- `core_regression_model.json` - Core AWS regression coefficients
- `cognito_regression_model.json` - Cognito hybrid model parameters

### Results
- `results.json` (per analysis) - Structured metrics and performance data
- `predictive_cost_analysis.csv` - Monthly projections
- Dashboard PNG files for visualization

## Error Handling

### Data Quality Validation
- **R² threshold**: Warnings for R² < 0.7
- **Data sufficiency**: Alerts for < 300 days (core) or < 6 months (Cognito)
- **Outlier detection**: Automated removal of anomalous cost data points

### Proxy Management
- **Port conflict detection**: Pre-check for occupied ports
- **Process cleanup**: Automatic TSH proxy termination
- **Fail-fast**: Script termination on proxy setup failure

## Performance Considerations

### Regression Computation
- **Normal equation**: Direct matrix computation for small datasets
- **Fallback**: Pseudoinverse for singular matrices
- **Complexity**: O(n³) for feature matrix inversion

### Memory Usage
- **Daily data**: ~365 records × 4 fields = minimal memory footprint
- **JSON export**: Compact model representation
- **CSV processing**: Pandas streaming for large datasets

## Future Improvements

### Potential Enhancements
- **Time series analysis**: ARIMA models for seasonal patterns
- **Feature engineering**: Moving averages, trend components
- **Cross-validation**: K-fold validation for model robustness
- **Ensemble methods**: Multiple model combination for improved accuracy

### Monitoring
- **Model drift detection**: Performance degradation alerts
- **Data quality metrics**: Automated data completeness scoring
- **Prediction confidence intervals**: Uncertainty quantification