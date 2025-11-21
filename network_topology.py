import time
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import info


class NetworkTopology:
    """Manages Mininet network topology and operations"""
    
    def __init__(self, controller_port=6653, topo_type = 'star'):
        self.topo_type = topo_type
        self.controller_port = controller_port
        self.net = None
        self.hosts = {}
        self.switch = None
    
    def create(self):
        info(f'*** Creating {self.topo_type} topology\n')
        self.net = Mininet(controller=RemoteController)
        self.net.addController('c0', port=self.controller_port)

        if self.topo_type == 'star':
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
            
        elif self.topo_type == 'tree':
            # Core Switch
            s1 = self.net.addSwitch('s1')
            self.switch = s1  # <--- FIX: Assign to class attribute for capture
            
            # Edge Switches
            s2 = self.net.addSwitch('s2')
            s3 = self.net.addSwitch('s3')
            
            # Links (Trunk)
            self.net.addLink(s1, s2)
            self.net.addLink(s1, s3)
            
            # Hosts
            h1 = self.net.addHost('h1', ip='10.0.0.1/24')
            h2 = self.net.addHost('h2', ip='10.0.0.2/24')
            h3 = self.net.addHost('h3', ip='10.0.0.3/24')
            h4 = self.net.addHost('h4', ip='10.0.0.4/24')
            
            self.hosts = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4}
            
            # Connect Dept A to Switch 2
            self.net.addLink(h1, s2)
            self.net.addLink(h2, s2)
            
            # Connect Dept B to Switch 3
            self.net.addLink(h3, s3)
            self.net.addLink(h4, s3)
        
    
    def start(self):
        info('*** Starting network\n')
        self.net.start()
        time.sleep(5)
        info('*** Testing connectivity\n')
        self.net.pingAll()
    
    def start_capture(self, output_file='./traffic_capture.pcap'):
        info(f'*** Starting packet capture on Core Switch: {output_file}\n')
        # Capture on "any" interface to see all traffic passing through the switch
        self.switch.cmd(f'tcpdump -i any -w {output_file} > /dev/null 2>&1 &')
        time.sleep(1)
    
    def stop_capture(self):
        info('*** Stopping packet capture\n')
        self.switch.cmd('killall tcpdump')
        time.sleep(2)
    
    def stop(self):
        info('*** Stopping network\n')
        if self.net:
            self.net.stop()
    
    def get_hosts(self):
        return list(self.hosts.values())
    
