#!/usr/bin/env python3

import json
import csv
from datetime import datetime
import sys

def convert_timestamp_to_month(timestamp_ms):
    """Convert millisecond timestamp to YYYY-MM format"""
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m')

def aggregate_daily_to_monthly(daily_data, aggregate_type='sum'):
    """Aggregate daily data to monthly totals or averages"""
    monthly_dict = {}

    if 'results' in daily_data and 'A' in daily_data['results']:
        frames = daily_data['results']['A']['frames']
        if frames and len(frames) > 0:
            timestamps = frames[0]['data']['values'][0]
            values = frames[0]['data']['values'][1]

            # Group daily values by month
            monthly_values = {}
            for ts, val in zip(timestamps, values):
                if val is not None:  # Skip null values
                    month = convert_timestamp_to_month(ts)
                    if month not in monthly_values:
                        monthly_values[month] = []
                    monthly_values[month].append(float(val))

            # Aggregate to monthly values
            for month, values_list in monthly_values.items():
                if aggregate_type == 'sum':
                    monthly_dict[month] = sum(values_list)
                elif aggregate_type == 'avg':
                    monthly_dict[month] = sum(values_list) / len(values_list)
                elif aggregate_type == 'last':
                    monthly_dict[month] = values_list[-1]  # Last value of the month

    return monthly_dict

def main():
    """Merge daily Grafana data (aggregated to monthly) and AWS data into a single CSV"""

    try:
        # Load daily transaction volume data
        with open('output/daily_transaction_volume_query.json', 'r') as f:
            transaction_data = json.load(f)

        # Load daily customer volume data
        with open('output/daily_customer_volume_query.json', 'r') as f:
            customer_data = json.load(f)

        # Load AWS Cognito cost data
        with open('output/aws_monthly_cognito_costs.json', 'r') as f:
            aws_data = json.load(f)

        # Aggregate daily data to monthly
        # Transaction volume: sum daily increases to get monthly totals
        transaction_dict = aggregate_daily_to_monthly(transaction_data, 'sum')

        # Customer volume: take last value of each month (customer counts are cumulative)
        customer_dict = aggregate_daily_to_monthly(customer_data, 'last')

        # Extract AWS Cognito cost data
        aws_dict = {}
        if 'ResultsByTime' in aws_data:
            for result in aws_data['ResultsByTime']:
                # Extract month from time period (e.g., "2024-10-01" -> "2024-10")
                start_date = result['TimePeriod']['Start']
                month = start_date[:7]  # "YYYY-MM"
                cost = float(result['Total']['AmortizedCost']['Amount'])
                aws_dict[month] = cost

        # Get all unique months and sort them
        all_months = sorted(set(transaction_dict.keys()) | set(customer_dict.keys()) | set(aws_dict.keys()))

        # Write merged CSV
        with open('output/cognito_cost_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['month', 'monthly_transaction_volume', 'monthly_customer_volume', 'cognito_cost'])

            for month in all_months:
                transaction_vol = transaction_dict.get(month, '')
                customer_vol = customer_dict.get(month, '')
                cognito_cost = aws_dict.get(month, '')
                writer.writerow([month, transaction_vol, customer_vol, cognito_cost])

        print(f"Monthly CSV file created with {len(all_months)} rows")
        print("Columns: month, monthly_transaction_volume, monthly_customer_volume, cognito_cost")
        print("Data aggregation: transactions=sum(daily), customers=last(daily), costs=monthly")

    except Exception as e:
        print(f"Error creating monthly CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()