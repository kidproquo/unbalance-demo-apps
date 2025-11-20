#!/bin/bash
# Test script for synchronized unbalance detection across all 3 approaches
#
# This script demonstrates how all three approaches can process the same data
# in real-time using Redis Streams for coordination.
#
# Prerequisites:
#   1. Redis server running: docker run -p 6379:6379 redis:7-alpine
#   2. Python environment with all dependencies installed
#   3. Dataset downloaded and extracted

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Synchronized Unbalance Detection Test${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo

# Check if Redis is running
echo -e "${YELLOW}Checking Redis connection...${NC}"
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running!${NC}"
    echo -e "${YELLOW}Start Redis with: docker run -p 6379:6379 redis:7-alpine${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Redis is running${NC}"
echo

# Configuration
MAX_WINDOWS=${1:-20}  # Default to 20 windows if not specified
TIME_WINDOW=10         # 10-second windows
LOG_INTERVAL=5         # Log every 5 windows

echo -e "${BLUE}Configuration:${NC}"
echo "  Max Windows: $MAX_WINDOWS"
echo "  Time Window: $TIME_WINDOW seconds"
echo "  Log Interval: Every $LOG_INTERVAL windows"
echo

# Create log directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Clean up old stream data
echo -e "${YELLOW}Cleaning up old Redis data...${NC}"
redis-cli DEL windows > /dev/null 2>&1 || true
redis-cli DEL detectors > /dev/null 2>&1 || true
echo -e "${GREEN}✓ Redis stream cleaned${NC}"
echo

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Starting synchronized processing...${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo

# Start data coordinator in background
echo -e "${YELLOW}Starting data coordinator...${NC}"
python data_coordinator.py \
    --max-windows $MAX_WINDOWS \
    --time-window $TIME_WINDOW \
    --normal-weight 0.9 \
    --redis-stream windows \
    > logs/coordinator_${TIMESTAMP}.log 2>&1 &
COORDINATOR_PID=$!
echo -e "${GREEN}✓ Coordinator started (PID: $COORDINATOR_PID)${NC}"

# Give coordinator a moment to start
sleep 2

# Start Approach 1 (CNN) in background
echo -e "${YELLOW}Starting Approach 1 (CNN)...${NC}"
python approach_1_cnn/approach_1_cnn.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-group detectors \
    --consumer-name cnn \
    --log-interval $LOG_INTERVAL \
    --dataset all \
    > logs/cnn_${TIMESTAMP}.log 2>&1 &
CNN_PID=$!
echo -e "${GREEN}✓ CNN approach started (PID: $CNN_PID)${NC}"

# Start Approach 2 (FFT) in background
echo -e "${YELLOW}Starting Approach 2 (FFT FCN)...${NC}"
python approach_2_fft/approach_2_fft.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-group detectors \
    --consumer-name fft \
    --log-interval $LOG_INTERVAL \
    --dataset all \
    > logs/fft_${TIMESTAMP}.log 2>&1 &
FFT_PID=$!
echo -e "${GREEN}✓ FFT approach started (PID: $FFT_PID)${NC}"

# Start Approach 3 (RFC) in background
echo -e "${YELLOW}Starting Approach 3 (Minimal RFC)...${NC}"
python approach_3_minimal_rfc/approach_3_minimal_rfc.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-group detectors \
    --consumer-name rfc \
    --log-interval $LOG_INTERVAL \
    --dataset all \
    > logs/rfc_${TIMESTAMP}.log 2>&1 &
RFC_PID=$!
echo -e "${GREEN}✓ RFC approach started (PID: $RFC_PID)${NC}"

echo
echo -e "${GREEN}All services started!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "${YELLOW}Monitoring progress...${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo -e "${YELLOW}Stopping all processes...${NC}"
    kill $COORDINATOR_PID $CNN_PID $FFT_PID $RFC_PID 2>/dev/null || true
    wait $COORDINATOR_PID $CNN_PID $FFT_PID $RFC_PID 2>/dev/null || true
    echo -e "${GREEN}✓ All processes stopped${NC}"
    echo
    echo -e "${BLUE}Logs saved to:${NC}"
    echo "  logs/coordinator_${TIMESTAMP}.log"
    echo "  logs/cnn_${TIMESTAMP}.log"
    echo "  logs/fft_${TIMESTAMP}.log"
    echo "  logs/rfc_${TIMESTAMP}.log"
    echo
    echo -e "${BLUE}Performance reports saved to:${NC}"
    echo "  figures/detections/"
}

trap cleanup EXIT INT TERM

# Wait for coordinator to finish
wait $COORDINATOR_PID

# Give approaches time to finish processing remaining windows
echo -e "${YELLOW}Waiting for approaches to finish processing...${NC}"
sleep 5

echo
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}Synchronized processing complete!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo

# Automatically trigger cleanup
exit 0
