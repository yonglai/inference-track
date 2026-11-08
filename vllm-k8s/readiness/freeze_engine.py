#!/usr/bin/env python3
"""Simulate a hung vLLM engine: freeze (SIGSTOP) or thaw (SIGCONT) the EngineCore process.

Run inside the pod:  python3 /experiment/freeze_engine.py freeze|thaw [--all]
Refuses to act unless exactly one process matches 'EngineCore' (pass --all to override).
"""
import os
import signal
import sys


def find():
    hits = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            comm = open(f"/proc/{d}/comm").read().strip()
        except OSError:
            continue
        if "EngineCore" in cmd or "EngineCore" in comm:
            hits.append((int(d), cmd or comm))
    return hits


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("freeze", "thaw"):
        print(__doc__)
        return 2
    sig = signal.SIGSTOP if sys.argv[1] == "freeze" else signal.SIGCONT
    hits = find()
    for pid, name in hits:
        print(f"match pid={pid}: {name[:100]}")
    if not hits or (len(hits) != 1 and "--all" not in sys.argv):
        print("refusing: expected exactly one EngineCore process (use --all to override)")
        return 1
    for pid, _ in hits:
        os.kill(pid, sig)
        print(f"sent {sig.name} to {pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
