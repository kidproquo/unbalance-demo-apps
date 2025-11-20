#!/usr/bin/env python3
"""
MCP Server for FFT FCN Unbalance Detection Monitoring

Provides real-time access to:
- Performance metrics (TP/FP/TN/FN, Accuracy, Precision, Recall)
- Latest unbalance detection events
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("FFT FCN Unbalance Detection Monitor")

# Default paths (can be overridden)
DETECTIONS_DIR = Path("../figures/detections")
DETECTIONS_LOG = None


def set_detections_dir(path: str):
    """Set the detections directory path (for integration with main script)."""
    global DETECTIONS_DIR, DETECTIONS_LOG
    DETECTIONS_DIR = Path(path)
    DETECTIONS_LOG = DETECTIONS_DIR / "detections.jsonl"


def get_latest_performance_report() -> Optional[dict]:
    """Find and parse the latest performance report file."""
    if not DETECTIONS_DIR.exists():
        return None

    # Find all performance report files
    report_files = list(DETECTIONS_DIR.glob("performance_report_*.txt"))
    if not report_files:
        return None

    # Get the most recent report
    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)

    # Parse the report
    with open(latest_report, 'r') as f:
        lines = f.readlines()

    # Extract metrics
    metrics = {
        'report_file': latest_report.name,
        'last_updated': datetime.fromtimestamp(latest_report.stat().st_mtime).isoformat(),
        'datasets': {},
        'overall': {}
    }

    # Parse the table
    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith('Dataset'):
            in_table = True
            continue
        if in_table and line.startswith('-'):
            continue
        if in_table and line.startswith('Overall'):
            # Parse overall metrics
            parts = line.split()
            if len(parts) >= 9:
                metrics['overall'] = {
                    'total': int(parts[1]),
                    'TP': int(parts[2]),
                    'FP': int(parts[3]),
                    'TN': int(parts[4]),
                    'FN': int(parts[5]),
                    'accuracy': float(parts[6]),
                    'precision': float(parts[7]),
                    'recall': float(parts[8])
                }
            break
        if in_table and line:
            # Parse dataset metrics
            parts = line.split()
            if len(parts) >= 9 and parts[0] in ['0E', '1E', '2E', '3E', '4E']:
                metrics['datasets'][parts[0]] = {
                    'total': int(parts[1]),
                    'TP': int(parts[2]),
                    'FP': int(parts[3]),
                    'TN': int(parts[4]),
                    'FN': int(parts[5]),
                    'accuracy': float(parts[6]),
                    'precision': float(parts[7]),
                    'recall': float(parts[8])
                }

    return metrics


def get_latest_detections(limit: int = 10) -> list:
    """Read the latest unbalance detection events from the log file."""
    if DETECTIONS_LOG is None or not DETECTIONS_LOG.exists():
        return []

    detections = []
    with open(DETECTIONS_LOG, 'r') as f:
        for line in f:
            if line.strip():
                detections.append(json.loads(line))

    # Return most recent detections
    return detections[-limit:]


@mcp.tool()
def get_performance_metrics() -> dict:
    """
    Get the latest performance metrics from the FFT FCN unbalance detection system.

    Returns a dictionary containing:
    - Per-dataset metrics (TP, FP, TN, FN, Accuracy, Precision, Recall)
    - Overall metrics across all datasets
    - Report file name and last update timestamp
    """
    metrics = get_latest_performance_report()
    if not metrics:
        return {
            "status": "no_data",
            "message": "No performance report found. The detection system may not be running."
        }

    return {
        "status": "success",
        "metrics": metrics
    }


@mcp.tool()
def get_recent_detections(limit: int = 10) -> dict:
    """
    Get the most recent unbalance detection events.

    Args:
        limit: Maximum number of detections to return (default: 10)

    Returns a dictionary containing:
    - List of recent detection events with timestamps, dataset info, and prediction scores
    - Count of total detections available
    """
    detections = get_latest_detections(limit)

    if not detections:
        return {
            "status": "no_data",
            "message": "No detection events found. The system may not have detected any unbalance yet.",
            "detections": []
        }

    return {
        "status": "success",
        "count": len(detections),
        "detections": detections
    }


@mcp.tool()
def get_system_status() -> dict:
    """
    Get the overall status of the FFT FCN unbalance detection system.

    Returns information about:
    - Whether the system is actively running
    - Latest performance metrics summary
    - Recent detection activity
    """
    metrics = get_latest_performance_report()
    recent_detections = get_latest_detections(limit=5)

    if not metrics:
        return {
            "status": "inactive",
            "message": "No performance report found. The detection system may not be running.",
            "running": False
        }

    # Check if report is recent (within last hour)
    last_update = datetime.fromisoformat(metrics['last_updated'])
    age_minutes = (datetime.now() - last_update).total_seconds() / 60

    return {
        "status": "active" if age_minutes < 60 else "stale",
        "running": age_minutes < 60,
        "last_update": metrics['last_updated'],
        "age_minutes": round(age_minutes, 1),
        "overall_accuracy": metrics['overall'].get('accuracy', 0.0) if metrics.get('overall') else 0.0,
        "total_windows_processed": metrics['overall'].get('total', 0) if metrics.get('overall') else 0,
        "recent_detections_count": len(recent_detections),
        "report_file": metrics['report_file']
    }


def run_mcp_server(transport: str = "sse", port: int = 8000):
    """
    Run the MCP server. Can be called from main script or standalone.

    Args:
        transport: Transport type - "sse" (HTTP/SSE, default) or "stdio" (for Claude Desktop)
        port: Port number for SSE transport (default: 8000)
    """
    if transport == "sse":
        # Run with HTTP/SSE transport
        mcp.run(transport="sse", port=port)
    else:
        # Run with stdio transport (for Claude Desktop)
        mcp.run()


if __name__ == "__main__":
    # Run the MCP server
    run_mcp_server()
