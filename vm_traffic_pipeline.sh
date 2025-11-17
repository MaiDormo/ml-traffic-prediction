#!/bin/bash

echo "=== VM Traffic Collection Pipeline ==="

DURATION=${1:-60} 
PATTERN=${2:-periodic} 
OUTPUT_LOCATION=${3-./traffic_capture.pcap}

echo "Duration: ${DURATION}s, Pattern: ${PATTERN}"

# Clean old data
echo "[1/4] Cleaning old data..." 
rm -f $OUTPUT_LOCATION traffic_features.csv

# Start controller
echo "[2/4] Starting SDN controller..."
ryu-manager sdn_controller.py > ryu.log 2>&1 &
RYU_PID=$!
sleep 5

if ! ps -p $RYU_PID > /dev/null; then
    echo "ERROR: Controller failed to start"
    exit 1
fi

# Run network simulation
echo "[3/4] Running network simulation..."
sudo python3 network_simulator.py $DURATION $PATTERN $OUTPUT_LOCATION

# Stop controller
echo "[4/4] Stopping controller..."
kill -TERM $RYU_PID 2>/dev/null
sleep 2

# Process capture
if [ -f /tmp/traffic_capture.pcap ]; then
    sudo chown $USER:$USER $OUTPUT_LOCATION
    PACKETS=$(sudo tcpdump -r $OUTPUT_LOCATION 2>/dev/null | wc -l)
    echo "✓ Captured $PACKETS packets"
    
    python3 pcap_preprocessor.py  $OUTPUT_LOCATION
    
    if [ -f traffic_features.csv ]; then
        echo "✓ Pipeline complete!"
        echo "Next: python3 train_models.py"
    fi
else
    echo "ERROR: Capture file not found"
    exit 1
fi
