from typing import Dict, List, Optional

import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader


FEATURE_COLUMNS = [
    "packets_per_sec",
    "bytes_per_sec",
    "syn_per_sec",
    "unique_dst_ports",
    "unique_dst_ips",
]


def _empty_window_rows(total_windows: int) -> List[Dict[str, float]]:
    return [
        {
            "window_start": window_start,
            "packets_per_sec": 0.0,
            "bytes_per_sec": 0.0,
            "syn_per_sec": 0.0,
            "unique_dst_ports": 0.0,
            "unique_dst_ips": 0.0,
        }
        for window_start in range(total_windows)
    ]


def extract_features(
    pcap_path: str,
    window_s: int = 1,
    total_windows: Optional[int] = None,
) -> pd.DataFrame:
    buckets: Dict[int, Dict[str, object]] = {}
    min_timestamp = None

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue

            pkt_timestamp = float(pkt.time)
            if min_timestamp is None:
                min_timestamp = int(pkt_timestamp)

            window_index = int(pkt_timestamp) - min_timestamp
            if total_windows is not None and (window_index < 0 or window_index >= total_windows):
                continue

            if window_index not in buckets:
                buckets[window_index] = {
                    "window_start": window_index,
                    "packets": 0,
                    "bytes": 0,
                    "syn": 0,
                    "dst_ports": set(),
                    "dst_ips": set(),
                }

            bucket = buckets[window_index]
            bucket["packets"] += 1
            bucket["bytes"] += len(pkt)

            if TCP in pkt:
                if pkt[TCP].flags & 0x02:
                    bucket["syn"] += 1
                bucket["dst_ports"].add(int(pkt[TCP].dport))
            elif UDP in pkt:
                bucket["dst_ports"].add(int(pkt[UDP].dport))

            bucket["dst_ips"].add(pkt[IP].dst)

    if not buckets and total_windows is None:
        return pd.DataFrame(columns=["window_start", *FEATURE_COLUMNS])

    if total_windows is None:
        total_windows = max(buckets.keys(), default=-1) + 1

    rows = _empty_window_rows(total_windows)
    for window_index, bucket in buckets.items():
        rows[window_index] = {
            "window_start": bucket["window_start"],
            "packets_per_sec": bucket["packets"] / window_s,
            "bytes_per_sec": bucket["bytes"] / window_s,
            "syn_per_sec": bucket["syn"] / window_s,
            "unique_dst_ports": float(len(bucket["dst_ports"])),
            "unique_dst_ips": float(len(bucket["dst_ips"])),
        }

    return pd.DataFrame(rows)
