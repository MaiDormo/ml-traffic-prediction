from mininet.log import setLogLevel, info
import sys

from network_topology import NetworkTopology
from traffic_patterns import PATTERNS


def run_simulation(duration=180, pattern='periodic', capture_file='/tmp/traffic_capture.pcap'):
    """
    Run network simulation with specified traffic pattern
    
    Args:
        duration: Simulation duration in seconds
        pattern: Traffic pattern to use (constant, periodic, stepped)
        capture_file: Path to save packet capture
    """
    # Create network topology
    topology = NetworkTopology(controller_port=6653)
    topology.create()
    topology.start()
    
    # Start packet capture
    topology.start_capture(capture_file)
    
    # Generate traffic
    info(f'\n*** Generating {pattern} traffic for {duration}s\n')
    pattern_class = PATTERNS[pattern]
    traffic_generator = pattern_class(hosts=topology.get_hosts(), duration=duration)
    traffic_generator.run()
    
    # Cleanup
    topology.stop_capture()
    topology.stop()
    
    info(f'*** Capture saved: {capture_file}\n')


def main():
    """Main entry point"""
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    pattern = sys.argv[2] if len(sys.argv) > 2 else 'periodic'
    capture_file = sys.argv[3] if len(sys.argv) > 3 else '/tmp/traffic_capture.pcap'
    
    if pattern not in PATTERNS:
        print(f'Error: Invalid pattern. Choose: {", ".join(PATTERNS.keys())}')
        sys.exit(1)
    
    info('='*60 + '\n')
    info('NETWORK TRAFFIC SIMULATOR\n')
    info('='*60 + '\n')
    info(f'Duration: {duration}s\n')
    info(f'Pattern: {pattern}\n')
    info(f'Capture: {capture_file}\n')
    info('='*60 + '\n')
    
    setLogLevel('info')
    run_simulation(duration, pattern, capture_file)


if __name__ == '__main__':
    main()