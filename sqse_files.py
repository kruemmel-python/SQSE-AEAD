#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqse_files.py
=============

Dateiverschlüsselung via SQSE (CC_OpenCL.dll).

Subcommands:
- encrypt  <infile> <outfile> [--K ... --steps ... --gpu ... (--key-seed S | --key-npy key.npy) [--xor]]
- decrypt  <infile> <outfile> [--gpu ... (--key-seed S | --key-npy key.npy) [--xor]]

Modi:
- Legacy-Container (Default): Header + float32-Blöcke (theta, p_mask) -> nicht bitgenau (FP32-Rundungen).
- XOR-Modus (--xor): Erzeuge Keystream über SQSE und XOR’e mit Bytes -> **bitgenau** (Hash==Hash).

Headerformate:
- Legacy: MAGIC="SQSE"(4s) | ver(uint8=1) | steps(uint8) | K(float32) | n_bytes(uint64) | [theta(float32*n)] | [p_mask(float32*n)]
- XOR   : MAGIC="SXOR"(4s) | ver(uint8=1) | steps(uint8) | K(float32) | n_bytes(uint64) | [cipher_bytes(n)]

RC-Konvention deiner DLL:
- initialize_gpu: Erfolg = 1
- sqse_load_kernels / encrypt / decrypt: Erfolg = 0
"""

from __future__ import annotations
import argparse
import struct
import os
from ctypes import cdll, c_int, c_float, c_char_p, POINTER
from pathlib import Path
from typing import Optional
import numpy as np

MAGIC_LEGACY = b"SQSE"
MAGIC_XOR    = b"SXOR"
VERSION = 1

class SQSEFileError(Exception): ...


# --------------------------- ctypes/Array-Helfer -----------------------------
def as_ptr(a: np.ndarray):
    return a.ctypes.data_as(POINTER(c_float))

def ensure_f32_c(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


# --------------------------- DLL Wrapper -------------------------------------
class SQSEDLL:
    def __init__(self, path: str, gpu_index: int):
        path = self._resolve_path(path)
        try:
            self.lib = cdll.LoadLibrary(path)
        except OSError as e:
            raise SQSEFileError(f"DLL konnte nicht geladen werden: {e}")
        # initialize_gpu (Erfolg == 1)
        if hasattr(self.lib, "initialize_gpu"):
            self.lib.initialize_gpu.argtypes = [c_int]
            self.lib.initialize_gpu.restype  = c_int
            rc = self.lib.initialize_gpu(gpu_index)
            if rc != 1:
                raise SQSEFileError(f"initialize_gpu({gpu_index}) RC={rc} (erwartet 1)")
        # bind SQSE
        self.lib.sqse_load_kernels.argtypes = [c_char_p]
        self.lib.sqse_load_kernels.restype  = c_int
        rc = self.lib.sqse_load_kernels(b"embedded")
        if rc != 0:
            raise SQSEFileError(f"sqse_load_kernels RC={rc}")

        self.lib.execute_sqse_encrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float), POINTER(c_float)
        ]
        self.lib.execute_sqse_encrypt_float.restype  = c_int

        self.lib.execute_sqse_decrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float)
        ]
        self.lib.execute_sqse_decrypt_float.restype  = c_int

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    # Vektor-APIs
    def encrypt_vec(self, x: np.ndarray, key: np.ndarray, K: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
        x   = ensure_f32_c(x)
        key = ensure_f32_c(key)
        if x.size != key.size:
            raise SQSEFileError(f"encrypt_vec: x.size != key.size ({x.size} != {key.size})")
        n = x.size
        theta = np.empty(n, dtype=np.float32)
        pmask = np.empty(n, dtype=np.float32)
        rc = self.lib.execute_sqse_encrypt_float(as_ptr(x), as_ptr(key), c_int(n), c_float(K), c_int(steps),
                                                 as_ptr(theta), as_ptr(pmask))
        if rc != 0:
            raise SQSEFileError(f"Encrypt RC={rc}")
        return theta, pmask

    def decrypt_vec(self, theta: np.ndarray, pmask: np.ndarray, key: np.ndarray, K: float, steps: int) -> np.ndarray:
        theta = ensure_f32_c(theta)
        pmask = ensure_f32_c(pmask)
        key   = ensure_f32_c(key)
        if not (theta.size == pmask.size == key.size):
            raise SQSEFileError(f"decrypt_vec: Größeninkonsistenz ({theta.size}/{pmask.size}/{key.size})")
        n = theta.size
        rec = np.empty(n, dtype=np.float32)
        rc = self.lib.execute_sqse_decrypt_float(as_ptr(theta), as_ptr(pmask), as_ptr(key),
                                                 c_int(n), c_float(K), c_int(steps), as_ptr(rec))
        if rc != 0:
            raise SQSEFileError(f"Decrypt RC={rc}")
        return rec


# --------------------------- Key-Erzeugung -----------------------------------
def make_key(n: int, key_seed: Optional[int], key_npy: Optional[Path]) -> np.ndarray:
    """float32-Key in [0,1) der Länge n."""
    if key_npy is not None:
        arr = np.load(str(key_npy))
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if arr.size < n:
            raise SQSEFileError(f"--key-npy enthält nur {arr.size} Werte, benötigt: {n}")
        return arr[:n].copy()
    if key_seed is None:
        raise SQSEFileError("Bitte --key-seed oder --key-npy angeben.")
    rng = np.random.default_rng(int(key_seed))
    return rng.random(n, dtype=np.float32)


# --------------------------- XOR-Keystream -----------------------------------
def keystream_from_sqse(n: int, dll: SQSEDLL, key: np.ndarray, K: float, steps: int) -> np.ndarray:
    """
    Erzeuge einen deterministischen Byte-Keystream der Länge n.
    Vorgehen:
    - Neutrales Inputsignal x=zeros (float32)
    - SQSE-Encrypt => (theta, pmask)
    - Keystream-Byte = floor(frac(theta)*256)  (frac(theta) := theta - floor(theta))
    Hinweis: Beide Seiten generieren denselben Keystream -> XOR ist bitgenau.
    """
    x = np.zeros(n, dtype=np.float32)
    theta, _ = dll.encrypt_vec(x, key, K, steps)
    frac = theta - np.floor(theta)            # frac in [0,1)
    ks = np.floor(frac * 256.0).astype(np.uint8)
    return ks


# --------------------------- Datei Encrypt/Decrypt ---------------------------
def encrypt_file_legacy(in_path: Path, out_path: Path, dll: SQSEDLL, K: float, steps: int, key: np.ndarray) -> None:
    data = in_path.read_bytes()
    x = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
    n = x.size
    theta, pmask = dll.encrypt_vec(x, key, K, steps)
    header = struct.pack("<4sBBfQ", MAGIC_LEGACY, VERSION, steps, K, n)
    out_path.write_bytes(header + theta.tobytes() + pmask.tobytes())

def decrypt_file_legacy(in_path: Path, out_path: Path, dll: SQSEDLL, key: np.ndarray) -> None:
    blob = in_path.read_bytes()
    hdr_size = struct.calcsize("<4sBBfQ")
    if len(blob) < hdr_size:
        raise SQSEFileError("Cipher zu kurz / kein Header.")
    magic, ver, steps, K, n = struct.unpack("<4sBBfQ", blob[:hdr_size])
    if magic != MAGIC_LEGACY: raise SQSEFileError("Kein SQSE-Header (Legacy).")
    if ver != VERSION:        raise SQSEFileError(f"Unbekannte Version {ver}")
    floats = n * 4
    if len(blob) < hdr_size + 2 * floats:
        raise SQSEFileError("Cipher-Blöcke unvollständig.")
    theta = np.frombuffer(blob[hdr_size: hdr_size + floats], dtype=np.float32).copy()
    pmask = np.frombuffer(blob[hdr_size + floats: hdr_size + 2 * floats], dtype=np.float32).copy()
    rec = dll.decrypt_vec(theta, pmask, key, K, steps)
    out_bytes = np.clip(np.round(rec * 255.0), 0, 255).astype(np.uint8).tobytes()
    out_path.write_bytes(out_bytes)

def encrypt_file_xor(in_path: Path, out_path: Path, dll: SQSEDLL, K: float, steps: int, key: np.ndarray) -> None:
    """
    Bitgenau: Plain XOR Keystream.
    Cipher = Header(SXOR, meta) || (plaintext_bytes XOR keystream)
    """
    pt = in_path.read_bytes()
    n = len(pt)
    ks = keystream_from_sqse(n, dll, key, K, steps)
    ct = np.frombuffer(pt, dtype=np.uint8) ^ ks
    header = struct.pack("<4sBBfQ", MAGIC_XOR, VERSION, steps, K, n)
    out_path.write_bytes(header + ct.tobytes())

def decrypt_file_xor(in_path: Path, out_path: Path, dll: SQSEDLL, key: np.ndarray) -> None:
    blob = in_path.read_bytes()
    hdr_size = struct.calcsize("<4sBBfQ")
    if len(blob) < hdr_size:
        raise SQSEFileError("Cipher zu kurz / kein Header.")
    magic, ver, steps, K, n = struct.unpack("<4sBBfQ", blob[:hdr_size])
    if magic != MAGIC_XOR: raise SQSEFileError("Kein SXOR-Header.")
    if ver != VERSION:     raise SQSEFileError(f"Unbekannte Version {ver}")
    ct = np.frombuffer(blob[hdr_size:], dtype=np.uint8)
    if ct.size != n:
        raise SQSEFileError("Cipher-Länge inkonsistent.")
    ks = keystream_from_sqse(n, dll, key, K, steps)
    pt = (ct ^ ks).tobytes()
    out_path.write_bytes(pt)


# --------------------------- CLI --------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sqse_files",
        description="Dateiverschlüsselung via SQSE (CC_OpenCL.dll). Nutze --xor für bitgenaue XOR-Verschlüsselung."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-d", "--dll", default="CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll")
    common.add_argument("-g", "--gpu", type=int, default=0, help="GPU-Index (Default: 0)")
    common.add_argument("--xor", action="store_true", help="XOR-Keystream-Modus (bitgenau) statt Legacy-Container")

    pe = sub.add_parser("encrypt", parents=[common], help="Datei verschlüsseln")
    pe.add_argument("infile", type=Path)
    pe.add_argument("outfile", type=Path)
    pe.add_argument("--K", type=float, default=1.2, help="chaos_K (Default: 1.2)")
    pe.add_argument("--steps", type=int, default=16, help="Iterationsanzahl (Default: 16)")
    g = pe.add_mutually_exclusive_group(required=True)
    g.add_argument("--key-seed", type=int, help="Deterministischer Seed für Key-Vector")
    g.add_argument("--key-npy", type=Path, help="Pfad zu .npy mit float32-Key-Werten")

    pd = sub.add_parser("decrypt", parents=[common], help="Datei entschlüsseln")
    pd.add_argument("infile", type=Path)
    pd.add_argument("outfile", type=Path)
    g2 = pd.add_mutually_exclusive_group(required=True)
    g2.add_argument("--key-seed", type=int, help="Deterministischer Seed (wie beim Encrypt)")
    g2.add_argument("--key-npy", type=Path, help="Pfad zu .npy (wie beim Encrypt)")

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    try:
        # Key vorbereiten (Länge aus Datei / Header abgeleitet)
        if args.cmd == "encrypt":
            data = Path(args.infile).read_bytes()
            n = len(data)
            # Key-Vektor für SQSE/Keystream
            if args.key_npy is not None:
                key = np.load(str(args.key_npy)).astype(np.float32).ravel()
                if key.size < n:
                    raise SQSEFileError(f"--key-npy hat nur {key.size} Werte, benötigt: {n}")
                key = key[:n].copy()
            else:
                rng = np.random.default_rng(int(args.key_seed))
                key = rng.random(n, dtype=np.float32)

            dll = SQSEDLL(args.dll, args.gpu)
            if args.xor:
                encrypt_file_xor(Path(args.infile), Path(args.outfile), dll, args.K, args.steps, key)
            else:
                # Legacy-Container (nicht bitgenau)
                x = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
                encrypt_file_legacy(Path(args.infile), Path(args.outfile), dll, args.K, args.steps, key)
            print(f"[ok] Encrypted: {args.infile} -> {args.outfile} (GPU={args.gpu}, K={args.K}, steps={args.steps}, xor={args.xor})")

        elif args.cmd == "decrypt":
            # Header lesen, um n festzustellen (Legacy/XOR)
            blob = Path(args.infile).read_bytes()
            hdr_size = struct.calcsize("<4sBBfQ")
            if len(blob) < hdr_size:
                raise SQSEFileError("Cipher zu kurz / kein Header.")
            magic, ver, steps, K, n = struct.unpack("<4sBBfQ", blob[:hdr_size])
            if magic not in (MAGIC_LEGACY, MAGIC_XOR):
                raise SQSEFileError("Unbekannter Header.")

            # Key-Vektor gleicher Länge n
            if args.key_npy is not None:
                key = np.load(str(args.key_npy)).astype(np.float32).ravel()
                if key.size < n:
                    raise SQSEFileError(f"--key-npy hat nur {key.size} Werte, benötigt: {n}")
                key = key[:n].copy()
            else:
                rng = np.random.default_rng(int(args.key_seed))
                key = rng.random(n, dtype=np.float32)

            dll = SQSEDLL(args.dll, args.gpu)
            if magic == MAGIC_XOR or args.xor:
                decrypt_file_xor(Path(args.infile), Path(args.outfile), dll, key)
            else:
                decrypt_file_legacy(Path(args.infile), Path(args.outfile), dll, key)
            print(f"[ok] Decrypted: {args.infile} -> {args.outfile} (GPU={args.gpu})")

        else:
            p.error("Unbekannter Befehl.")

    except SQSEFileError as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
