#!/usr/bin/env python3
from pathlib import Path
import struct
import sys
import os

# Minimal allowed Argon2 parameters (configurable via env)
MIN_ARGON_MEM_MB = int(os.getenv("MIN_ARGON_MEM_MB", "256"))
MIN_ARGON_T = int(os.getenv("MIN_ARGON_T", "2"))
MIN_ARGON_P = int(os.getenv("MIN_ARGON_P", "1"))

MAGIC = b"SX3A"
VERSION_EXPECTED = 2

def read_header_v2(path: Path):
    with path.open("rb") as fh:
        base_len = struct.calcsize("<4sBBHfIIQHHIII")
        base = fh.read(base_len)
        if len(base) < base_len:
            raise RuntimeError("Header zu kurz.")
        magic, ver, flags, _res, K, steps, chunk_bytes, total_bytes, slen, nlen, amem, atime, apar = struct.unpack("<4sBBHfIIQHHIII", base)
        if magic != MAGIC or ver != VERSION_EXPECTED:
            raise RuntimeError("Unbekanntes Format oder Version.")
        return int(amem), int(atime), int(apar)

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_header.py <file.aead>")
        sys.exit(2)
    p = Path(sys.argv[1])
    try:
        amem, atime, apar = read_header_v2(p)
    except Exception as e:
        print("ERROR reading header:", e)
        sys.exit(3)
    print(f"Found Argon2 params: mem={amem}MiB, t={atime}, p={apar}")
    ok = True
    if amem < MIN_ARGON_MEM_MB:
        print(f"mem too small: {amem} < {MIN_ARGON_MEM_MB}")
        ok = False
    if atime < MIN_ARGON_T:
        print(f"time cost too small: {atime} < {MIN_ARGON_T}")
        ok = False
    if apar < MIN_ARGON_P:
        print(f"parallelism too small: {apar} < {MIN_ARGON_P}")
        ok = False
    if not ok:
        print("Header validation FAILED.")
        sys.exit(4)
    print("Header validation OK.")
    sys.exit(0)

if __name__ == "__main__":
    main()
