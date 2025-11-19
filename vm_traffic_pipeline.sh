#!/bin/bash

echo "=== VM Traffic Collection Pipeline ==="

DURATION=${1:-60} 
PATTERN=${2:-web} 
OUTPUT_LOCATION=${3-/tmp/traffic_capture.pcap}
TOPO_TYPE=${4:-tree}  # Added support for topology argument

echo "Duration: ${DURATION}s, Pattern: ${PATTERN}, Output: ${OUTPUT_LOCATION}, Topo: ${TOPO_TYPE}"

# Clean old data AND Mininet artifacts
echo "[1/5] Cleaning old data..." 
sudo mn -c > /dev/null 2>&1  # <--- ADD THIS LINE (suppress output to keep logs clean)
rm -f $OUTPUT_LOCATION traffic_features.csv

# Start controller
echo "[2/5] Starting SDN controller..."
ryu-manager sdn_controller.py > ryu.log 2>&1 &
RYU_PID=$!
sleep 5

if ! ps -p $RYU_PID > /dev/null; then
    echo "ERROR: Controller failed to start"
    exit 1
fi

# Run network simulation
echo "[3/5] Running network simulation..."
# Pass the TOPO_TYPE variable to the python script
sudo python3 network_simulator.py $DURATION $PATTERN $OUTPUT_LOCATION $TOPO_TYPE

# Stop controller
echo "[4/5] Stopping controller..."
kill -TERM $RYU_PID 2>/dev/null
sleep 2

# Moving file in correct location
echo "[5/5] Moving pcap to directory..."
# 1. Take ownership of the file (change root -> current user)
sudo chown $USER:$USER $OUTPUT_LOCATION

# 2. Now move it safely
mv $OUTPUT_LOCATION ~/ml-traffic-prediction/
