import sys
from mininet.log import setLogLevel, info
from network_topology import NetworkTopology
from traffic_patterns import PATTERNS

def run_simulation(duration, pattern, capture_file, topo_type):
    # Pass topo_type to NetworkTopology
    topology = NetworkTopology(controller_port=6653, topo_type=topo_type)
    topology.create()
    topology.start()
    
    topology.start_capture(capture_file)
    
    info(f'\n*** Generating {pattern} traffic on {topo_type} topology for {duration}s\n')
    
    # Traffic Generation
    if pattern in PATTERNS:
        pattern_class = PATTERNS[pattern]
        # Pass all hosts to the pattern
        traffic_generator = pattern_class(hosts=topology.get_hosts(), duration=duration)
        traffic_generator.run()
    else:
        info(f"Pattern {pattern} not found, skipping traffic.\n")
    
    topology.stop_capture()
    topology.stop()
    info(f'*** Capture saved: {capture_file}\n')

def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    pattern = sys.argv[2] if len(sys.argv) > 2 else 'periodic'
    capture_file = sys.argv[3] if len(sys.argv) > 3 else './traffic_capture.pcap'
    topo_type = sys.argv[4] if len(sys.argv) > 4 else 'star' 
    
    info('='*60 + '\n')
    info(f'NETWORK SIMULATOR: {topo_type.upper()} Topology | Pattern: {pattern}\n')
    info('='*60 + '\n')
    
    setLogLevel('info')
    run_simulation(duration, pattern, capture_file, topo_type)

if __name__ == '__main__':
    main()