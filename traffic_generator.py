import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from scapy.layers.inet import IP, UDP, TCP
from scapy.packet import Packet
from scapy.utils import PcapWriter


@dataclass
class Phase:
    name: str
    duration_s: int
    label: int  # 0 normal, 1 attack
    profile: str  # normal, burst, udp_flood, syn_flood


DEFAULT_PHASES = [
    Phase("normal_1", 60, 0, "normal"),
    Phase("firmware_burst", 10, 0, "burst"),
    Phase("normal_2", 20, 0, "normal"),
    Phase("udp_flood", 20, 1, "udp_flood"),
    Phase("normal_3", 10, 0, "normal"),
    Phase("syn_flood", 20, 1, "syn_flood"),
    Phase("normal_4", 20, 0, "normal"),
]


def _packets_for_second(profile: str, src_ip: str, dst_ip: str, second_index: int) -> List[Packet]:
    packets: List[Packet] = []

    # Keep the normal traffic low-rate and small.
    if profile == "normal":
        count = random.randint(1, 3)
        for _ in range(count):
            dport = random.choice([53, 123, 80])
            payload = bytes([0x42]) * random.randint(20, 60)
            packets.append(IP(src=src_ip, dst=dst_ip) / UDP(dport=dport, sport=random.randint(20000, 60000)) / payload)

    # Firmware update burst: more packets and larger payloads, still benign.
    elif profile == "burst":
        count = random.randint(20, 40)
        for _ in range(count):
            dport = 443
            payload = bytes([0x99]) * random.randint(400, 900)
            packets.append(IP(src=src_ip, dst=dst_ip) / UDP(dport=dport, sport=random.randint(20000, 60000)) / payload)

    elif profile == "udp_flood":
        count = random.randint(200, 300)
        for _ in range(count):
            dport = 80
            payload = bytes([0xAB]) * random.randint(100, 300)
            packets.append(IP(src=src_ip, dst=dst_ip) / UDP(dport=dport, sport=random.randint(10000, 60000)) / payload)

    elif profile == "syn_flood":
        count = random.randint(150, 250)
        for _ in range(count):
            dport = random.randint(1, 1024)
            sport = random.randint(1024, 65535)
            packets.append(IP(src=src_ip, dst=dst_ip) / TCP(dport=dport, sport=sport, flags="S"))

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return packets


def generate_pcap(
    output_pcap: str,
    labels_csv: str,
    phases: List[Phase] = None,
    src_ip: str = "10.0.0.2",
    dst_ip: str = "10.0.0.1",
    seed: int = 1337,
) -> None:
    if phases is None:
        phases = DEFAULT_PHASES

    random.seed(seed)

    total_seconds = sum(p.duration_s for p in phases)
    labels: List[Tuple[int, int, str]] = []

    base_time = time.time()
    current_second = 0

    writer = PcapWriter(output_pcap, append=False, sync=True)
    try:
        for phase in phases:
            for _ in range(phase.duration_s):
                labels.append((current_second, phase.label, phase.name))
                packets = _packets_for_second(phase.profile, src_ip, dst_ip, current_second)
                for pkt in packets:
                    pkt.time = base_time + current_second + random.random()
                    writer.write(pkt)
                current_second += 1
    finally:
        writer.close()

    with open(labels_csv, "w", encoding="utf-8") as f:
        f.write("window_start,label,phase\n")
        for window_start, label, phase_name in labels:
            f.write(f"{window_start},{label},{phase_name}\n")


if __name__ == "__main__":
    generate_pcap("pcaps/iot_traffic.pcap", "results/labels.csv")
