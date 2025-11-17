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
    """Periodic traffic with sine wave pattern (scaled period based on duration)"""
    
    def run(self):
        info('[Periodic] Video stream: adaptive quality with smooth oscillation\n')
        
        self.h1.cmd('iperf -s -u -p 5002 > /dev/null 2>&1 &')
        time.sleep(2)
        
        self.start_time = time.time()
        
        # Scale period and amplitude based on simulation duration
        if self.duration > 300:  # > 5 minutes
            period = 120  # 2-minute cycles for longer runs
            amplitude = 1.5  # Reduced oscillation: 5 ± 1.5 Mbps (3.5-6.5 Mbps)
            base = 5
            jitter_range = 0.15  # ±0.15 Mbps
        elif self.duration > 120:  # > 2 minutes
            period = 60  # 1-minute cycles
            amplitude = 2  # 5 ± 2 Mbps (3-7 Mbps)
            base = 5
            jitter_range = 0.2
        else:  # Short simulations (original behavior)
            period = 40  # 40-second cycles
            amplitude = 3  # 5 ± 3 Mbps (2-8 Mbps)
            base = 5
            jitter_range = 0.3
        
        info(f'  Period: {period}s, Range: {base-amplitude:.1f}-{base+amplitude:.1f} Mbps\n')
        
        while self.remaining() > 0:
            # Sine wave with scaled parameters
            phase = (self.elapsed() % period) / period * 2 * math.pi
            base_rate = base + amplitude * math.sin(phase)
            jitter = random.uniform(-jitter_range, jitter_range)
            bitrate = max(2, base_rate + jitter)
            
            segment = min(3, self.remaining())
            
            # Log every 10 seconds for long runs, every iteration for short runs
            if self.duration <= 120 or int(self.elapsed()) % 10 == 0:
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


class RandomBurstTraffic(TrafficPattern):
    """Highly randomized bursts with unpredictable idle gaps"""

    def run(self):
        info('[RandomBurst] Chaotic bursts with idle gaps\n')

        self.h1.cmd('iperf -s -u -p 5005 > /dev/null 2>&1 &')
        time.sleep(2)

        self.start_time = time.time()

        while self.remaining() > 0:
            burst_duration = random.uniform(1.0, 4.0)
            idle_duration = random.uniform(0.5, 3.0)
            burst_rate = random.uniform(2, 9)  # Mbps

            # Occasionally trigger a heavy burst
            if random.random() < 0.15:
                burst_rate = random.uniform(10, 15)
                burst_duration = min(self.remaining(), random.uniform(2.0, 6.0))
                info(f'  ⚡ Heavy burst: {burst_rate:.1f} Mbps for {burst_duration:.1f}s\n')
            else:
                info(f'  Burst: {burst_rate:.1f} Mbps for {burst_duration:.1f}s\n')

            burst_duration = min(burst_duration, self.remaining())

            if burst_duration > 0:
                self.h2.popen(
                    f'iperf -c {self.h1.IP()} -u -p 5005 -b {burst_rate}M -t {int(burst_duration)} '
                    '> /dev/null 2>&1', shell=True
                ).wait()

            # Random idle/jitter period (simulate silence or background noise)
            idle_duration = min(idle_duration, self.remaining())
            if idle_duration > 0:
                noise_rate = random.uniform(0.2, 1.0)
                info(f'    Idle jitter: {noise_rate:.1f} Mbps background for {idle_duration:.1f}s\n')
                self.h3.popen(
                    f'iperf -c {self.h1.IP()} -u -p 5005 -b {noise_rate}M -t {int(idle_duration)} '
                    '> /dev/null 2>&1', shell=True
                ).wait()


# Pattern registry
PATTERNS = {
    'constant': ConstantTraffic,
    'periodic': PeriodicTraffic,
    'stepped': SteppedTraffic,
    'random': RandomBurstTraffic
}
