#!/usr/bin/env python3
"""
Analyze time differences between consecutive unbalance detections.
"""

import json
from datetime import datetime
from collections import defaultdict

approaches = ['cnn', 'fft', 'rfc']
results = {}

for approach in approaches:
    path = f'outputs/{approach}/detections.jsonl'

    # Load all detections with timestamps
    detections = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Only process Level 1 detections (since that's what we're currently testing)
                if data['dataset'] == '1E':
                    timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
                    detections.append({
                        'timestamp': timestamp,
                        'window_idx': data['window_idx']
                    })

    # Sort by timestamp
    detections.sort(key=lambda x: x['timestamp'])

    # Calculate time differences between consecutive detections
    time_diffs = []
    for i in range(1, len(detections)):
        diff = (detections[i]['timestamp'] - detections[i-1]['timestamp']).total_seconds()
        time_diffs.append(diff)

    results[approach] = {
        'total_detections': len(detections),
        'time_diffs': time_diffs,
        'first_detection': detections[0]['timestamp'] if detections else None,
        'last_detection': detections[-1]['timestamp'] if detections else None
    }

print("=" * 80)
print("Time Interval Analysis Between Consecutive Level 1 Detections")
print("=" * 80)
print()

for approach in ['cnn', 'fft', 'rfc']:
    data = results[approach]
    time_diffs = data['time_diffs']

    if not time_diffs:
        print(f"{approach.upper()}: No consecutive detections found")
        continue

    avg_diff = sum(time_diffs) / len(time_diffs)
    min_diff = min(time_diffs)
    max_diff = max(time_diffs)
    median_diff = sorted(time_diffs)[len(time_diffs) // 2]

    # Calculate total time span
    if data['first_detection'] and data['last_detection']:
        total_span = (data['last_detection'] - data['first_detection']).total_seconds()
    else:
        total_span = 0

    print(f"{'='*80}")
    print(f"{approach.upper()} - Level 1 Detection Timing")
    print(f"{'='*80}")
    print(f"Total Level 1 detections: {data['total_detections']}")
    print(f"First detection: {data['first_detection']}")
    print(f"Last detection:  {data['last_detection']}")
    print(f"Total time span: {total_span:.1f} seconds ({total_span/60:.1f} minutes)")
    print()
    print(f"Time between consecutive detections:")
    print(f"  Average:  {avg_diff:.1f} seconds ({avg_diff/60:.2f} minutes)")
    print(f"  Median:   {median_diff:.1f} seconds ({median_diff/60:.2f} minutes)")
    print(f"  Minimum:  {min_diff:.1f} seconds ({min_diff/60:.2f} minutes)")
    print(f"  Maximum:  {max_diff:.1f} seconds ({max_diff/60:.2f} minutes)")
    print()

    # Distribution analysis
    ranges = [
        (0, 10, "0-10s (rapid-fire)"),
        (10, 30, "10-30s (quick succession)"),
        (30, 60, "30-60s (moderate gap)"),
        (60, 120, "1-2 min (longer gap)"),
        (120, 300, "2-5 min (sparse)"),
        (300, float('inf'), "5+ min (very sparse)")
    ]

    print("Distribution of time gaps:")
    for min_t, max_t, label in ranges:
        count = sum(1 for d in time_diffs if min_t <= d < max_t)
        pct = (count / len(time_diffs) * 100) if time_diffs else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<25} {count:4d} ({pct:5.1f}%) {bar}")
    print()

# Comparison summary
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

comparison_data = []
for approach in ['cnn', 'fft', 'rfc']:
    data = results[approach]
    if data['time_diffs']:
        avg_diff = sum(data['time_diffs']) / len(data['time_diffs'])
        comparison_data.append((approach.upper(), avg_diff, data['total_detections']))

comparison_data.sort(key=lambda x: x[1])

print(f"{'Approach':<10} {'Avg Time Between Detections':<30} {'Total Detections':<20}")
print("-" * 80)
for approach, avg_time, total in comparison_data:
    print(f"{approach:<10} {avg_time:6.1f}s ({avg_time/60:5.2f} min)                 {total:<20}")

print()
print("KEY INSIGHTS:")
print()

if comparison_data:
    fastest = comparison_data[0]
    slowest = comparison_data[-1]

    print(f"• {fastest[0]} detects most frequently: every {fastest[1]:.1f}s on average")
    print(f"• {slowest[0]} detects least frequently: every {slowest[1]:.1f}s on average")
    print(f"• Detection frequency difference: {slowest[1]/fastest[1]:.2f}x")
    print()

    # Calculate detection rate per hour
    print("Detection rates (assuming continuous operation):")
    for approach, avg_time, total in comparison_data:
        rate_per_hour = 3600 / avg_time if avg_time > 0 else 0
        print(f"  {approach}: {rate_per_hour:.1f} detections/hour")
    print()

print("=" * 80)
