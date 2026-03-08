from typing import Dict, List

import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader


def extract_features(pcap_path: str, window_s: int = 1) -> pd.DataFrame:
    buckets: Dict[int, Dict[str, object]] = {}

    start_time = None
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            if start_time is None:
                start_time = pkt.time
            window_index = int((pkt.time - start_time) // window_s)
            if window_index not in buckets:
                buckets[window_index] = {
                    "window_start": window_index * window_s,
                    "packets": 0,
                    "bytes": 0,
                    "syn": 0,
                    "dst_ports": set(),
                    "dst_ips": set(),
                }
            b = buckets[window_index]
            b["packets"] += 1
            b["bytes"] += len(pkt)

            if TCP in pkt:
                if pkt[TCP].flags & 0x02:  # SYN
                    b["syn"] += 1
                b["dst_ports"].add(int(pkt[TCP].dport))
            elif UDP in pkt:
                b["dst_ports"].add(int(pkt[UDP].dport))
            b["dst_ips"].add(pkt[IP].dst)

    if not buckets:
        return pd.DataFrame(columns=[
            "window_start",
            "packets_per_sec",
            "bytes_per_sec",
            "syn_per_sec",
            "unique_dst_ports",
            "unique_dst_ips",
        ])

    rows: List[Dict[str, object]] = []
    for window_index in sorted(buckets.keys()):
        b = buckets[window_index]
        rows.append({
            "window_start": b["window_start"],
            "packets_per_sec": b["packets"] / window_s,
            "bytes_per_sec": b["bytes"] / window_s,
            "syn_per_sec": b["syn"] / window_s,
            "unique_dst_ports": len(b["dst_ports"]),
            "unique_dst_ips": len(b["dst_ips"]),
        })

    return pd.DataFrame(rows)
