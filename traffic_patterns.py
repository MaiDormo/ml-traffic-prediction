import time
import math
import random
from mininet.log import info


class TrafficPattern:
    """Base class for traffic generation patterns"""
    
    def __init__(self, hosts, duration):
        self.h1, self.h2, self.h3, self.h4 = hosts
        self.duration = duration
        self.start_time = None
    
    def run(self):
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def elapsed(self):
        """Get elapsed time since start"""
        return time.time() - self.start_time if self.start_time else 0
    
    def remaining(self):
        """Get remaining time"""
        return max(0, self.duration - self.elapsed())


class ConstantTraffic(TrafficPattern):
    """Constant bitrate with minor fluctuations (±10%)"""
    
    def run(self):
        info('[Constant] File transfer: 5 Mbps ±10% noise\n')
        
        self.h1.cmd('iperf -s -u -p 5001 > /dev/null 2>&1 &')
        time.sleep(2)
        
        self.start_time = time.time()
        while self.remaining() > 0:
            noise_factor = random.uniform(0.9, 1.1)
            bitrate = 5 * noise_factor
            segment = min(5, self.remaining())
            
            self.h2.popen(
                f'iperf -c {self.h1.IP()} -u -p 5001 -b {bitrate}M -t {int(segment)} '
                '> /dev/null 2>&1', shell=True
            ).wait()


class PeriodicTraffic(TrafficPattern):
    """Periodic traffic with sine wave pattern (40s period)"""
    
    def run(self):
        info('[Periodic] Video stream: 2-8 Mbps adaptive quality\n')
        
        self.h1.cmd('iperf -s -u -p 5002 > /dev/null 2>&1 &')
        time.sleep(2)
        
        self.start_time = time.time()
        period = 40  # seconds
        
        while self.remaining() > 0:
            # Sine wave: 5 ± 3 Mbps
            phase = (self.elapsed() % period) / period * 2 * math.pi
            base_rate = 5 + 3 * math.sin(phase)
            jitter = random.uniform(-0.3, 0.3)
            bitrate = max(2, base_rate + jitter)
            
            segment = min(3, self.remaining())
            info(f'  t={self.elapsed():.0f}s: {bitrate:.1f} Mbps\n')
            
            self.h2.popen(
                f'iperf -c {self.h1.IP()} -u -p 5002 -b {bitrate}M -t {int(segment)} '
                '> /dev/null 2>&1', shell=True
            ).wait()


class SteppedTraffic(TrafficPattern):
    """Stepped traffic simulating daily usage (7 phases)"""
    
    PHASES = [
        (2, 'Early Morning'),
        (4, 'Morning Peak'),
        (3, 'Midday'),
        (5, 'Afternoon'),
        (8, 'Evening Peak'),
        (6, 'Late Evening'),
        (2, 'Night')
    ]
    
    def run(self):
        info('[Stepped] Daily pattern: 7 phases\n')
        
        self.h1.cmd('iperf -s -u -p 5004 > /dev/null 2>&1 &')
        time.sleep(2)
        
        phase_duration = self.duration / len(self.PHASES)
        
        for i, (mbps, label) in enumerate(self.PHASES, 1):
            info(f'  Phase {i}/{len(self.PHASES)}: {mbps} Mbps ({label})\n')
            
            phase_start = time.time()
            while time.time() - phase_start < phase_duration:
                # Add ±15% variation within phase
                variation = random.uniform(0.85, 1.15)
                bitrate = mbps * variation
                segment = min(3, phase_duration - (time.time() - phase_start))
                
                if segment > 0:
                    self.h2.popen(
                        f'iperf -c {self.h1.IP()} -u -p 5004 -b {bitrate}M -t {int(segment)} '
                        '> /dev/null 2>&1', shell=True
                    ).wait()


# Pattern registry
PATTERNS = {
    'constant': ConstantTraffic,
    'periodic': PeriodicTraffic,
    'stepped': SteppedTraffic
}
