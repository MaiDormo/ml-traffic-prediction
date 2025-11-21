import time
import math
import random
from mininet.log import info

class TrafficPattern:
    def __init__(self, hosts, duration):
        self.hosts = hosts # List of all hosts
        self.duration = duration
        self.start_time = None
    
    def run(self):
        raise NotImplementedError
    
    def elapsed(self):
        return time.time() - self.start_time if self.start_time else 0
    
    def remaining(self):
        return max(0, self.duration - self.elapsed())

class ConstantTraffic(TrafficPattern):
    def run(self):
        info('[Constant] File transfer: 5 Mbps ±10% noise\n')
        for h in self.hosts:
            h.cmd('iperf -s -u -p 5001 > /dev/null 2>&1 &')
        time.sleep(2)
        self.start_time = time.time()
        
        while self.remaining() > 0:
            src, dst = random.sample(self.hosts, 2)
            dst.cmd('iperf -s -u -p 5001 > /dev/null 2>&1 &') 
            src.popen(f'iperf -c {dst.IP()} -u -p 5001 -b 5M -t 2 > /dev/null 2>&1')
            time.sleep(1)

class PeriodicTraffic(TrafficPattern):
    def run(self):
        info('[Periodic] Mesh Sine Wave: Variable Load across all nodes\n')
        for h in self.hosts:
            h.cmd('iperf -s -u -p 5001 > /dev/null 2>&1 &')
        
        time.sleep(2)
        self.start_time = time.time()
        period = 60
        
        while self.remaining() > 0:
            phase = (self.elapsed() % period) / period * 2 * math.pi
            base_bw = 3.0 
            amplitude = 2.0
            current_bw = base_bw + amplitude * math.sin(phase)
            src, dst = random.sample(self.hosts, 2)
            segment = min(3, self.remaining())
            info(f'  t={self.elapsed():.0f}s | {src.name}->{dst.name} | {current_bw:.2f} Mbps\n')
            src.popen(f'iperf -c {dst.IP()} -u -p 5001 -b {current_bw}M -t {segment} > /dev/null 2>&1').wait()

class SteppedTraffic(TrafficPattern):
    PHASES = [(2, 'Early Morning'), (4, 'Morning Peak'), (3, 'Midday'), (5, 'Afternoon'), (8, 'Evening Peak'), (6, 'Late Evening'), (2, 'Night')]
    
    def run(self):
        info('[Stepped] Daily pattern: 7 phases\n')
        self.hosts[0].cmd('iperf -s -u -p 5004 > /dev/null 2>&1 &')
        time.sleep(2)
        phase_duration = self.duration / len(self.PHASES)
        self.start_time = time.time()
        
        for i, (mbps, label) in enumerate(self.PHASES, 1):
            info(f'  Phase {i}/{len(self.PHASES)}: {mbps} Mbps ({label})\n')
            phase_start = time.time()
            while time.time() - phase_start < phase_duration:
                variation = random.uniform(0.85, 1.15)
                bitrate = mbps * variation
                segment = min(3, phase_duration - (time.time() - phase_start))
                if segment > 0:
                    self.hosts[1].popen(f'iperf -c {self.hosts[0].IP()} -u -p 5004 -b {bitrate}M -t {int(segment)} > /dev/null 2>&1').wait()

class RandomBurstTraffic(TrafficPattern):
    def run(self):
        info('[RandomBurst] Chaotic bursts with idle gaps\n')
        self.hosts[0].cmd('iperf -s -u -p 5005 > /dev/null 2>&1 &')
        time.sleep(2)
        self.start_time = time.time()

        while self.remaining() > 0:
            burst_duration = random.uniform(1.0, 4.0)
            idle_duration = random.uniform(0.5, 3.0)
            burst_rate = random.uniform(2, 9)

            if random.random() < 0.15:
                burst_rate = random.uniform(10, 15)
                burst_duration = min(self.remaining(), random.uniform(2.0, 6.0))
                info(f'  ⚡ Heavy burst: {burst_rate:.1f} Mbps\n')
            else:
                info(f'  Burst: {burst_rate:.1f} Mbps\n')

            burst_duration = min(burst_duration, self.remaining())
            if burst_duration > 0:
                self.hosts[1].popen(f'iperf -c {self.hosts[0].IP()} -u -p 5005 -b {burst_rate}M -t {int(burst_duration)} > /dev/null 2>&1').wait()
            
            time.sleep(min(idle_duration, self.remaining()))

class WebBrowsingTraffic(TrafficPattern):
    def run(self):
        info('[Web] HTTP Browsing: h1 serves content, others download\n')
        server = self.hosts[0]
        clients = self.hosts[1:]
        server.cmd('echo "Welcome to the authentic web simulation" > index.html')
        server.cmd('python3 -m http.server 8000 &')
        time.sleep(2)
        self.start_time = time.time()
        
        while self.remaining() > 0:
            client = random.choice(clients)
            wait_time = random.uniform(0.5, 2.0)
            info(f'  t={self.elapsed():.0f}s | {client.name} requesting HTTP page...\n')
            client.cmd(f'wget -O /dev/null http://{server.IP()}:8000/index.html > /dev/null 2>&1 &')
            time.sleep(wait_time)

class DayCycleTraffic(TrafficPattern):
    """
    Simulates a full day cycle: Rise -> Peak -> Fall.
    Non-repetitive. Tests if model learns the 'Trend'.
    """
    def run(self):
        info('[DayCycle] Simulating Morning -> Peak -> Evening cycle\n')
        
        # Start servers on all hosts
        for h in self.hosts:
            h.cmd('iperf -s -u -p 5001 > /dev/null 2>&1 &')
        
        time.sleep(2)
        self.start_time = time.time()
        
        # We map the total duration to a 0 -> Pi cycle (0 to 180 degrees)
        # sin(0) = 0 (Morning), sin(Pi/2) = 1 (Noon), sin(Pi) = 0 (Night)
        
        while self.remaining() > 0:
            elapsed = self.elapsed()
            
            # Calculate position in the "day" (0.0 to 3.14)
            day_progress = (elapsed / self.duration) * math.pi
            
            # Intensity follows a bell curve (Sine wave half-cycle)
            # Base: 1Mbps, Peak: +8Mbps
            intensity = 1.0 + (8.0 * math.sin(day_progress))
            
            # Add randomness (Clouds/Noise) so it's not perfect
            jitter = random.uniform(-0.5, 0.5)
            current_bw = max(0.5, intensity + jitter)
            
            # Mesh Traffic: Random Source -> Random Dest
            src, dst = random.sample(self.hosts, 2)
            segment = min(2, self.remaining())
            if int(elapsed) % 10 == 0:
                 info(f'  t={elapsed:.0f}s | Day Phase: {day_progress/math.pi:.2f} | BW: {current_bw:.2f} Mbps\n')
            src.popen(
                f'iperf -c {dst.IP()} -u -p 5001 -b {current_bw}M -t {segment} '
                '> /dev/null 2>&1', shell=True
            ).wait()

PATTERNS = {
    'constant': ConstantTraffic,
    'periodic': PeriodicTraffic,
    'stepped': SteppedTraffic,
    'random': RandomBurstTraffic,
    'web': WebBrowsingTraffic,
    'daily': DayCycleTraffic
}