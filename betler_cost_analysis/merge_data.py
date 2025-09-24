#!/usr/bin/env python3

import json
import csv
from datetime import datetime
from collections import defaultdict
import sys

def convert_timestamp_to_date(timestamp_ms):
    """Convert millisecond timestamp to YYYY-MM-DD format"""
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')

def main():
    try:
        # Read transaction volume data
        with open('output/transaction_volume_query.json', 'r') as f:
            transaction_data = json.load(f)

        # Read customer volume data
        with open('output/customer_volume_query.json', 'r') as f:
            customer_data = json.load(f)

        # Read AWS cost data
        with open('output/aws_daily_costs.json', 'r') as f:
            aws_data = json.load(f)

        # Extract transaction volume data
        transaction_times = transaction_data['results']['A']['frames'][0]['data']['values'][0]
        transaction_values = transaction_data['results']['A']['frames'][0]['data']['values'][1]

        # Extract customer volume data
        customer_times = customer_data['results']['A']['frames'][0]['data']['values'][0]
        customer_values = customer_data['results']['A']['frames'][0]['data']['values'][1]

        # Create dictionaries for lookup
        transaction_dict = {}
        for i, timestamp in enumerate(transaction_times):
            date = convert_timestamp_to_date(timestamp)
            transaction_dict[date] = transaction_values[i]

        customer_dict = {}
        for i, timestamp in enumerate(customer_times):
            date = convert_timestamp_to_date(timestamp)
            customer_dict[date] = customer_values[i]

        # Extract AWS cost data
        aws_dict = {}
        for result in aws_data['ResultsByTime']:
            date = result['TimePeriod']['Start']
            amount = float(result['Total']['AmortizedCost']['Amount'])
            aws_dict[date] = amount

        # Get all unique dates and sort them
        all_dates = sorted(set(list(transaction_dict.keys()) + list(customer_dict.keys()) + list(aws_dict.keys())))

        # Write CSV file
        with open('output/cost_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['date', 'transaction_volume', 'customer_volume', 'aws_cost'])

            for date in all_dates:
                transaction_vol = transaction_dict.get(date, '')
                customer_vol = customer_dict.get(date, '')
                aws_cost = aws_dict.get(date, '')
                writer.writerow([date, transaction_vol, customer_vol, aws_cost])

        print(f"CSV file created with {len(all_dates)} rows")
        print("Columns: date, transaction_volume, customer_volume, aws_cost")

    except Exception as e:
        print(f"Error creating CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()