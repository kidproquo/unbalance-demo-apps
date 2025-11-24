#!/usr/bin/env python3
"""
Analyze Level 1 detection agreement across CNN, FFT, and RFC approaches.
"""

import json
from collections import defaultdict

# Load all detections from the three approaches
approaches = ['cnn', 'fft', 'rfc']
detections = {}

for approach in approaches:
    path = f'outputs/{approach}/detections.jsonl'
    detections[approach] = []

    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Only keep Level 1 detections
                    if data['dataset'] == '1E':
                        # Different approaches use different field names for prediction score
                        prediction = data.get('prediction') or data.get('mean_prediction') or data.get('max_prediction', 0.0)
                        detections[approach].append({
                            'window_idx': data['window_idx'],
                            'timestamp': data['timestamp'],
                            'prediction': prediction
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error in {approach} at line {line_num}: {e}")
                    print(f"  Line content: {line[:100]}")
                    continue

print(f"Level 1 detections per approach:")
print(f"  CNN: {len(detections['cnn'])}")
print(f"  FFT: {len(detections['fft'])}")
print(f"  RFC: {len(detections['rfc'])}")
print()

# Create a mapping of window_idx to which approaches detected it
window_detections = defaultdict(set)

for approach in approaches:
    for det in detections[approach]:
        window_idx = det['window_idx']
        window_detections[window_idx].add(approach)

# Count by number of approaches that detected each window
detected_by_all_three = []
detected_by_two = []
detected_by_one = []

for window_idx, detecting_approaches in window_detections.items():
    count = len(detecting_approaches)
    if count == 3:
        detected_by_all_three.append((window_idx, detecting_approaches))
    elif count == 2:
        detected_by_two.append((window_idx, detecting_approaches))
    elif count == 1:
        detected_by_one.append((window_idx, detecting_approaches))

print(f"=== Level 1 Detection Agreement Analysis ===")
print(f"Total unique Level 1 windows detected: {len(window_detections)}")
print()
print(f"Detected by all 3 approaches: {len(detected_by_all_three)}")
print(f"Detected by exactly 2 approaches: {len(detected_by_two)}")
print(f"Detected by exactly 1 approach: {len(detected_by_one)}")
print()

# Breakdown of 2-approach detections
if detected_by_two:
    two_approach_combos = defaultdict(int)
    for window_idx, approaches_set in detected_by_two:
        combo = '+'.join(sorted(approaches_set))
        two_approach_combos[combo] += 1

    print(f"Breakdown of 2-approach detections:")
    for combo, count in sorted(two_approach_combos.items()):
        print(f"  {combo}: {count}")
    print()

# Breakdown of 1-approach detections
if detected_by_one:
    one_approach_counts = defaultdict(int)
    for window_idx, approaches_set in detected_by_one:
        approach = list(approaches_set)[0]
        one_approach_counts[approach] += 1

    print(f"Breakdown of 1-approach detections:")
    for approach in sorted(one_approach_counts.keys()):
        print(f"  {approach} only: {one_approach_counts[approach]}")
    print()

# Show some examples from each category
print(f"=== Examples ===")
if detected_by_all_three:
    print(f"\nExample windows detected by all 3 (showing first 5):")
    for window_idx, _ in detected_by_all_three[:5]:
        # Get prediction scores from each approach
        cnn_pred = next((d['prediction'] for d in detections['cnn'] if d['window_idx'] == window_idx), None)
        fft_pred = next((d['prediction'] for d in detections['fft'] if d['window_idx'] == window_idx), None)
        rfc_pred = next((d['prediction'] for d in detections['rfc'] if d['window_idx'] == window_idx), None)
        cnn_str = f"{cnn_pred:.3f}" if cnn_pred is not None else "N/A"
        fft_str = f"{fft_pred:.3f}" if fft_pred is not None else "N/A"
        rfc_str = f"{rfc_pred:.3f}" if rfc_pred is not None else "N/A"
        print(f"  Window {window_idx}: CNN={cnn_str}, FFT={fft_str}, RFC={rfc_str}")

if detected_by_two:
    print(f"\nExample windows detected by exactly 2 (showing first 5):")
    for window_idx, approaches_set in detected_by_two[:5]:
        cnn_pred = next((d['prediction'] for d in detections['cnn'] if d['window_idx'] == window_idx), None)
        fft_pred = next((d['prediction'] for d in detections['fft'] if d['window_idx'] == window_idx), None)
        rfc_pred = next((d['prediction'] for d in detections['rfc'] if d['window_idx'] == window_idx), None)
        detected_by = ', '.join(sorted(approaches_set))
        cnn_str = f"{cnn_pred:.3f}" if cnn_pred is not None else "N/A"
        fft_str = f"{fft_pred:.3f}" if fft_pred is not None else "N/A"
        rfc_str = f"{rfc_pred:.3f}" if rfc_pred is not None else "N/A"
        print(f"  Window {window_idx} [{detected_by}]: CNN={cnn_str}, FFT={fft_str}, RFC={rfc_str}")

if detected_by_one:
    print(f"\nExample windows detected by only 1 (showing first 5):")
    for window_idx, approaches_set in detected_by_one[:5]:
        cnn_pred = next((d['prediction'] for d in detections['cnn'] if d['window_idx'] == window_idx), None)
        fft_pred = next((d['prediction'] for d in detections['fft'] if d['window_idx'] == window_idx), None)
        rfc_pred = next((d['prediction'] for d in detections['rfc'] if d['window_idx'] == window_idx), None)
        detected_by = list(approaches_set)[0]
        cnn_str = f"{cnn_pred:.3f}" if cnn_pred is not None else "N/A"
        fft_str = f"{fft_pred:.3f}" if fft_pred is not None else "N/A"
        rfc_str = f"{rfc_pred:.3f}" if rfc_pred is not None else "N/A"
        print(f"  Window {window_idx} [{detected_by} only]: CNN={cnn_str}, FFT={fft_str}, RFC={rfc_str}")
