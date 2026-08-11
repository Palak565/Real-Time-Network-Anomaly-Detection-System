"""
Captures live traffic via tshark (through pyshark), aggregates packets
into flows keyed by 5-tuple, and periodically writes finished flows as
newline-delimited JSON files into config.STREAM_INPUT_DIR, matching the
schema in config.BASIC_FEATURE_COLS. stream_processor.py watches that
directory and scores each new file with Spark Structured Streaming.

Requires tshark installed and on PATH (comes with Wireshark, or
`sudo apt install tshark`). On Linux you generally need to either run
this as root or add your user to the `wireshark` group for packet
capture permissions.

Run:
    python capture_producer.py --interface eth0
    (use `tshark -D` to list interface names/numbers on your machine)

NOTE ON ACCURACY: `flag` (SF/S0/REJ/...) and `service` are approximated
here with simple heuristics -- they will not always match NSL-KDD's
original labelling exactly. Good enough for a demo / for showing the
pipeline works end-to-end; call this out as a known limitation in your
writeup rather than presenting it as ground truth.
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone

import pyshark

import config

SERVICE_MAP = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain_u", 80: "http", 110: "pop_3", 143: "imap4",
    443: "http_443", 3306: "sql_net", 8080: "http_8080",
}


def map_service(port):
    if port in SERVICE_MAP:
        return SERVICE_MAP[port]
    return "private" if port and port >= 1024 else "other"


class FlowTracker:
    """Aggregates packets into flows keyed by (src_ip, sport, dst_ip, dport, proto)."""

    def __init__(self, timeout=5):
        self.flows = {}
        self.timeout = timeout
        self.lock = threading.Lock()

    @staticmethod
    def _key_for(pkt):
        try:
            proto = pkt.transport_layer
            if proto is None or not hasattr(pkt, "ip"):
                return None
            layer = pkt[proto]
            return (pkt.ip.src, int(layer.srcport), pkt.ip.dst, int(layer.dstport), proto.lower())
        except AttributeError:
            return None

    def update(self, pkt):
        key = self._key_for(pkt)
        if key is None:
            return
        try:
            now = float(pkt.sniff_timestamp)
            length = int(pkt.length)
        except (AttributeError, ValueError):
            return

        with self.lock:
            flow = self.flows.get(key)
            if flow is None:
                flow = {
                    "start": now, "last": now,
                    "src_ip": key[0], "sport": key[1],
                    "dst_ip": key[2], "dport": key[3],
                    "protocol_type": key[4],
                    "src_bytes": 0, "dst_bytes": 0,
                    "syn": 0, "fin": 0, "rst": 0, "ack": 0, "urg": 0,
                    "wrong_fragment": 0,
                }
                self.flows[key] = flow

            flow["last"] = now
            if pkt.ip.src == flow["src_ip"]:
                flow["src_bytes"] += length
            else:
                flow["dst_bytes"] += length

            if hasattr(pkt, "ip") and getattr(pkt.ip, "flags_mf", "0") == "1":
                flow["wrong_fragment"] += 1

            if key[4] == "tcp" and hasattr(pkt, "tcp"):
                t = pkt.tcp
                if getattr(t, "flags_syn", "0") == "1":
                    flow["syn"] += 1
                if getattr(t, "flags_fin", "0") == "1":
                    flow["fin"] += 1
                if getattr(t, "flags_reset", "0") == "1":
                    flow["rst"] += 1
                if getattr(t, "flags_ack", "0") == "1":
                    flow["ack"] += 1
                if getattr(t, "flags_urg", "0") == "1":
                    flow["urg"] += 1

    @staticmethod
    def _flag_for(flow):
        if flow["rst"] > 0:
            return "REJ" if flow["syn"] > 0 and flow["ack"] == 0 else "RSTR"
        if flow["syn"] > 0 and flow["fin"] > 0 and flow["ack"] > 0:
            return "SF"
        if flow["syn"] > 0 and flow["fin"] == 0:
            return "S0"
        return "OTH"

    def pop_expired(self):
        now = time.time()
        expired = []
        with self.lock:
            for key in list(self.flows.keys()):
                if now - self.flows[key]["last"] > self.timeout:
                    expired.append(self.flows.pop(key))
        return expired

    def to_record(self, flow):
        duration = max(0.0, flow["last"] - flow["start"])
        return {
            "duration": duration,
            "protocol_type": flow["protocol_type"],
            "service": map_service(flow["dport"]),
            "flag": self._flag_for(flow),
            "src_bytes": flow["src_bytes"],
            "dst_bytes": flow["dst_bytes"],
            "land": 1 if flow["src_ip"] == flow["dst_ip"] and flow["sport"] == flow["dport"] else 0,
            "wrong_fragment": flow["wrong_fragment"],
            "urgent": flow["urg"],
            "src_ip": flow["src_ip"], "dst_ip": flow["dst_ip"],
            "sport": flow["sport"], "dport": flow["dport"],
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }


def run(interface, flush_interval=5, flow_timeout=5):
    os.makedirs(config.STREAM_INPUT_DIR, exist_ok=True)
    tracker = FlowTracker(timeout=flow_timeout)
    capture = pyshark.LiveCapture(interface=interface)

    def sniff_loop():
        for pkt in capture.sniff_continuously():
            tracker.update(pkt)

    threading.Thread(target=sniff_loop, daemon=True).start()
    print(f"Capturing on '{interface}'... flow batches -> {config.STREAM_INPUT_DIR}")

    while True:
        time.sleep(flush_interval)
        expired = tracker.pop_expired()
        if not expired:
            continue
        records = [tracker.to_record(flow) for flow in expired]
        fname = f"flows_{int(time.time() * 1000)}.json"
        path = os.path.join(config.STREAM_INPUT_DIR, fname)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} flows -> {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True, help="e.g. eth0, Wi-Fi, en0 (see `tshark -D`)")
    parser.add_argument("--flush-interval", type=int, default=5, help="seconds between writing batches")
    parser.add_argument("--flow-timeout", type=int, default=5, help="seconds of inactivity before a flow is considered done")
    args = parser.parse_args()
    run(args.interface, args.flush_interval, args.flow_timeout)
