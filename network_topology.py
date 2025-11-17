import time
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import info


class NetworkTopology:
    """Manages Mininet network topology and operations"""
    
    def __init__(self, controller_port=6653):
        self.controller_port = controller_port
        self.net = None
        self.hosts = {}
        self.switch = None
    
    def create(self):
        """Create 4-host star topology with OpenFlow switch"""
        info('*** Creating network topology\n')
        
        self.net = Mininet(controller=RemoteController)
        
        # Add controller
        self.net.addController('c0', port=self.controller_port)
        
        # Add hosts
        self.hosts['h1'] = self.net.addHost('h1', ip='10.0.0.1/24')
        self.hosts['h2'] = self.net.addHost('h2', ip='10.0.0.2/24')
        self.hosts['h3'] = self.net.addHost('h3', ip='10.0.0.3/24')
        self.hosts['h4'] = self.net.addHost('h4', ip='10.0.0.4/24')
        
        # Add switch
        self.switch = self.net.addSwitch('s1')
        
        # Create links (star topology)
        for host in self.hosts.values():
            self.net.addLink(host, self.switch)
        
        info(f'*** Topology: 4 hosts connected to 1 switch\n')
    
    def start(self):
        """Start network and verify connectivity"""
        info('*** Starting network\n')
        self.net.start()
        time.sleep(5)
        
        info('*** Testing connectivity\n')
        self.net.pingAll()
    
    def start_capture(self, output_file='/tmp/traffic_capture.pcap'):
        """Start packet capture on switch"""
        info(f'*** Starting packet capture: {output_file}\n')
        self.switch.cmd(f'tcpdump -i s1-eth1 -w {output_file} > /dev/null 2>&1 &')
        time.sleep(1)
    
    def stop_capture(self):
        """Stop packet capture"""
        info('*** Stopping packet capture\n')
        self.switch.cmd('killall tcpdump')
        time.sleep(2)
    
    def stop(self):
        """Stop network and cleanup"""
        info('*** Stopping network\n')
        if self.net:
            self.net.stop()
    
    def get_hosts(self):
        """Get list of host objects"""
        return [
            self.hosts['h1'],
            self.hosts['h2'],
            self.hosts['h3'],
            self.hosts['h4']
        ]
