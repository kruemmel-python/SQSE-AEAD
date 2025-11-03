#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqse_aead.py
============

Authenticated Encryption (AEAD) mit deiner CC_OpenCL.dll als Keystream-Generator.

Eigenschaften
-------------
- Bitgenau (XOR-Keystream über SQSE), Chunk-Streaming (Default 8 MiB)
- AEAD: BLAKE3-MAC (32 Byte) über (Header || Cipher) → Integrität & Authentizität
- Schlüsselmaterial:
    * Passwort → Argon2id (standardmäßig m=512 MiB, t=3, p=4)
    * ODER Seed (--seed) → 256-bit Key (für Reproduzierbarkeit)
    * ODER Rohmaterial aus .npy (--key-npy)
- Nonce (16 B) pro Datei + Counter pro Chunk → Keystream-Reuse verhindert
- SQSE-Parameter: chaos_K, steps (im Header gespeichert)

Headerformat (SX3A v1)
----------------------
Little-Endian:
  magic(4s)="SX3A" |
  ver(u8)=1 |
  flags(u8) (bit0: 1=AEAD, bit1: 1=Argon2) |
  reserved(u16)=0 |
  chaos_K(float32) |
  steps(u32) |
  chunk_bytes(u32) |
  total_bytes(u64) |
  salt_len(u16) | nonce_len(u16)

Anschließend: salt (salt_len), nonce (nonce_len), danach Cipher-Bytes, am Ende MAC(32 B).

Rückgabecodes deiner DLL:
- initialize_gpu: Erfolg == 1
- sqse_load_kernels / encrypt / decrypt: Erfolg == 0

Abhängigkeiten:
- numpy, blake3 (pip install blake3), (optional) argon2-cffi für Passwortmodus
"""

from __future__ import annotations
import argparse
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ctypes import cdll, c_int, c_float, c_char_p, POINTER

try:
    from blake3 import blake3
except Exception as e:
    raise RuntimeError("Bitte 'pip install blake3' installieren.") from e

# Argon2 optional (nur wenn --pass genutzt wird)
try:
    from argon2.low_level import hash_secret_raw, Type as Argon2Type
    ARGON2_AVAILABLE = True
except Exception:
    ARGON2_AVAILABLE = False


# ------------------- Konstanten / Header -------------------
MAGIC = b"SX3A"
VERSION = 1

FLAG_AEAD   = 1 << 0
FLAG_ARGON2 = 1 << 1

MAC_LEN = 32      # BLAKE3
NONCE_LEN = 16    # 128-bit Nonce
SALT_LEN  = 16    # 128-bit Salt

DEFAULT_CHUNK = 8 * 1024 * 1024  # 8 MiB


# ------------------- Fehlerklasse -------------------
class SQSEAEADError(Exception): ...


# ------------------- ctypes/Array-Helpers -------------------
def as_ptr(a: np.ndarray):
    return a.ctypes.data_as(POINTER(c_float))

def ensure_f32_c(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


# ------------------- DLL-Wrapper -------------------
class SQSEDLL:
    def __init__(self, path: str, gpu_index: int):
        path = self._resolve_path(path)
        try:
            self.lib = cdll.LoadLibrary(path)
        except OSError as e:
            raise SQSEAEADError(f"DLL konnte nicht geladen werden: {e}")

        # initialize_gpu (Erfolg==1)
        if hasattr(self.lib, "initialize_gpu"):
            self.lib.initialize_gpu.argtypes = [c_int]
            self.lib.initialize_gpu.restype  = c_int
            rc = self.lib.initialize_gpu(gpu_index)
            if rc != 1:
                raise SQSEAEADError(f"initialize_gpu({gpu_index}) RC={rc} (erwartet 1)")

        # sqse_load_kernels (Erfolg==0)
        self.lib.sqse_load_kernels.argtypes = [c_char_p]
        self.lib.sqse_load_kernels.restype  = c_int
        rc = self.lib.sqse_load_kernels(b"embedded")
        if rc != 0:
            raise SQSEAEADError(f"sqse_load_kernels RC={rc}")

        # Encrypt
        self.lib.execute_sqse_encrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float), POINTER(c_float)
        ]
        self.lib.execute_sqse_encrypt_float.restype = c_int

        # Decrypt (für Keystream nicht nötig, aber vollständig)
        self.lib.execute_sqse_decrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float)
        ]
        self.lib.execute_sqse_decrypt_float.restype = c_int

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    def encrypt_vec(self, x: np.ndarray, key: np.ndarray, K: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        x   = ensure_f32_c(x)
        key = ensure_f32_c(key)
        if x.size != key.size:
            raise SQSEAEADError(f"x.size != key.size ({x.size} != {key.size})")
        n = x.size
        theta = np.empty(n, dtype=np.float32)
        pmask = np.empty(n, dtype=np.float32)
        rc = self.lib.execute_sqse_encrypt_float(as_ptr(x), as_ptr(key), c_int(n), c_float(K), c_int(steps),
                                                 as_ptr(theta), as_ptr(pmask))
        if rc != 0:
            raise SQSEAEADError(f"Encrypt RC={rc}")
        return theta, pmask


# ------------------- KDF & Keying -------------------
def kdf_from_password(password: str, salt: bytes, mem_mb: int, time_cost: int, parallelism: int) -> bytes:
    """
    Liefert 32-Byte Session-Key via Argon2id.
    """
    if not ARGON2_AVAILABLE:
        raise SQSEAEADError("Argon2 nicht verfügbar. Bitte 'pip install argon2-cffi' installieren oder --seed/--key-npy verwenden.")
    mem_kib = mem_mb * 1024
    key = hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=mem_kib,
        parallelism=parallelism,
        hash_len=32,
        type=Argon2Type.ID
    )
    return key  # 32 bytes


def kdf_from_seed(seed: int) -> bytes:
    """
    32-Byte Session-Key deterministisch aus Seed via BLAKE3.
    """
    h = blake3()
    h.update(b"SQSE-SEED-KEY\0")
    h.update(seed.to_bytes(16, "little", signed=False))
    return h.digest(length=32)


def kdf_from_npy(npy_path: Path) -> bytes:
    """
    32-Byte Session-Key aus .npy (beliebige Länge); BLAKE3-Hashing.
    """
    arr = np.load(str(npy_path))
    blob = memoryview(np.ascontiguousarray(arr)).tobytes()
    return blake3(blob).digest(length=32)


def expand_key_float32(n: int, ctx_key: bytes, nonce: bytes, counter: int) -> np.ndarray:
    """
    Erzeugt einen float32-Keyvektor in [0,1) der Länge n:
    - BLAKE3 XOF über (ctx_key || nonce || counter)
    - interpretiere als uint32 → / 2**32 → float32
    Vorteil: deterministisch, ohne RNG-Zustand.
    """
    # 4 Bytes pro float32
    need = n * 4
    hk = blake3(key=ctx_key)
    hk.update(b"SQSE-F32-EXPAND\0")
    hk.update(nonce)
    hk.update(counter.to_bytes(8, "little"))
    stream = hk.digest(length=need)
    u32 = np.frombuffer(stream, dtype=np.uint32, count=n)
    f = (u32.astype(np.float64) / (2**32)).astype(np.float32)
    return f


# ------------------- Keystream via SQSE --------------------------------------
def keystream_bytes(dll: SQSEDLL, n: int, chaos_K: float, steps: int, key_f32: np.ndarray) -> np.ndarray:
    """
    Erzeuge n Bytes Keystream:
    - neutraler Input x=zeros (float32)
    - SQSE-Encrypt → theta
    - frac(theta) * 256 → uint8
    """
    x = np.zeros(n, dtype=np.float32)
    theta, _ = dll.encrypt_vec(x, key_f32, chaos_K, steps)
    frac = theta - np.floor(theta)    # in [0,1)
    ks = np.floor(frac * 256.0).astype(np.uint8)
    return ks


# ------------------- Header I/O ---------------------------------------------
def write_header(fh, flags: int, K: float, steps: int, chunk_bytes: int,
                 total_bytes: int, salt: bytes, nonce: bytes) -> None:
    hdr = struct.pack("<4sBBHfIIQHH",
                      MAGIC, VERSION, flags, 0,
                      np.float32(K).item(), int(steps), int(chunk_bytes),
                      int(total_bytes), len(salt), len(nonce))
    fh.write(hdr)
    fh.write(salt)
    fh.write(nonce)

def read_header(fh) -> tuple[int, float, int, int, int, bytes, bytes]:
    base = fh.read(struct.calcsize("<4sBBHfIIQHH"))
    if len(base) == 0:
        raise SQSEAEADError("Leere Datei.")
    if len(base) < 4+1+1+2+4+4+4+8+2+2:
        raise SQSEAEADError("Header zu kurz.")
    magic, ver, flags, _reserved, K, steps, chunk_bytes, total_bytes, slen, nlen = \
        struct.unpack("<4sBBHfIIQHH", base)
    if magic != MAGIC or ver != VERSION:
        raise SQSEAEADError("Unbekanntes Format oder Version.")
    salt = fh.read(slen)
    nonce = fh.read(nlen)
    if len(salt) != slen or len(nonce) != nlen:
        raise SQSEAEADError("Header Felder unvollständig.")
    return flags, float(K), int(steps), int(chunk_bytes), int(total_bytes), salt, nonce


# ------------------- AEAD: MAC ----------------------------------------------
def mac_update(h: "blake3", flags: int, K: float, steps: int, chunk_bytes: int,
               total_bytes: int, salt: bytes, nonce: bytes) -> None:
    # Gleiche Reihenfolge wie Header
    h.update(MAGIC)
    h.update(bytes([VERSION]))
    h.update(bytes([flags]))
    h.update(b"\x00\x00")  # reserved
    h.update(np.float32(K).tobytes())
    h.update(struct.pack("<I", steps))
    h.update(struct.pack("<I", chunk_bytes))
    h.update(struct.pack("<Q", total_bytes))
    h.update(struct.pack("<H", len(salt)))
    h.update(struct.pack("<H", len(nonce)))
    h.update(salt)
    h.update(nonce)


# ------------------- Encrypt / Decrypt --------------------------------------
def encrypt_file(in_path: Path, out_path: Path, dll_path: str, gpu: int,
                 K: float, steps: int,
                 password: Optional[str], seed: Optional[int], key_npy: Optional[Path],
                 chunk_bytes: int, argon_mem_mb: int, argon_t: int, argon_p: int) -> None:
    pt_size = in_path.stat().st_size

    salt = os.urandom(SALT_LEN) if password else (b"" if (seed is not None or key_npy) else os.urandom(SALT_LEN))
    nonce = os.urandom(NONCE_LEN)

    # Session-Key herstellen (32 B)
    if password:
        key32 = kdf_from_password(password, salt, argon_mem_mb, argon_t, argon_p)
        flags = FLAG_AEAD | FLAG_ARGON2
    elif seed is not None:
        key32 = kdf_from_seed(int(seed))
        flags = FLAG_AEAD
    elif key_npy is not None:
        key32 = kdf_from_npy(key_npy)
        flags = FLAG_AEAD
    else:
        raise SQSEAEADError("Bitte --pass oder --seed oder --key-npy angeben.")

    dll = SQSEDLL(dll_path, gpu)

    # Output + Header + MAC vorbereiten
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        write_header(fout, flags, K, steps, chunk_bytes, pt_size, salt, nonce)

        mac = blake3(key=key32)
        mac_update(mac, flags, K, steps, chunk_bytes, pt_size, salt, nonce)

        remaining = pt_size
        chunk_idx = 0

        while remaining > 0:
            n = min(chunk_bytes, remaining)
            block = fin.read(n)
            if len(block) != n:
                raise SQSEAEADError("Kurzlese im Plaintext.")
            # Ableitung float32-Key pro Chunk
            key_f32 = expand_key_float32(n, key32, nonce, chunk_idx)
            ks = keystream_bytes(dll, n, K, steps, key_f32)
            ct = (np.frombuffer(block, dtype=np.uint8) ^ ks).tobytes()
            fout.write(ct)
            mac.update(ct)

            remaining -= n
            chunk_idx += 1

        # MAC ans Ende
        fout.write(mac.digest(length=MAC_LEN))


def decrypt_file(in_path: Path, out_path: Path, dll_path: str, gpu: int,
                 password: Optional[str], seed: Optional[int], key_npy: Optional[Path]) -> None:
    with open(in_path, "rb") as fin:
        flags, K, steps, chunk_bytes, total_bytes, salt, nonce = read_header(fin)

        # Session-Key rekonstruieren
        if flags & FLAG_ARGON2:
            if not password:
                raise SQSEAEADError("Datei erwartet Passwort/Argon2, aber --pass fehlt.")
            if not ARGON2_AVAILABLE:
                raise SQSEAEADError("Argon2 nicht verfügbar. Bitte 'pip install argon2-cffi'.")
            # Default Argon2-Parameter für Decrypt (müssen identisch sein!)
            # Wir versuchen hier Standardwerte; besser: im Header kodieren (könnte man erweitern).
            key32 = kdf_from_password(password, salt, 512, 3, 4)
        else:
            if seed is not None:
                key32 = kdf_from_seed(int(seed))
            elif key_npy is not None:
                key32 = kdf_from_npy(key_npy)
            else:
                raise SQSEAEADError("Datei erwartet Seed/NPY, aber Parameter fehlen.")

        dll = SQSEDLL(dll_path, gpu)

        # MAC prüfen: wir lesen Cipher (total_bytes) + MAC(32)
        cipher_and_mac = fin.read()
        if len(cipher_and_mac) < total_bytes + MAC_LEN:
            raise SQSEAEADError("Cipher/MAC unvollständig.")
        cipher = cipher_and_mac[:total_bytes]
        mac_stored = cipher_and_mac[total_bytes: total_bytes + MAC_LEN]

        mac = blake3(key=key32)
        mac_update(mac, flags, K, steps, chunk_bytes, total_bytes, salt, nonce)

        # Streaming-Entschlüsselung + MAC-Verify parallel
        with open(out_path, "wb") as fout:
            remaining = total_bytes
            chunk_idx = 0
            offset = 0
            while remaining > 0:
                n = min(chunk_bytes, remaining)
                ct = cipher[offset: offset + n]
                mac.update(ct)

                key_f32 = expand_key_float32(n, key32, nonce, chunk_idx)
                ks = keystream_bytes(dll, n, K, steps, key_f32)
                pt = (np.frombuffer(ct, dtype=np.uint8) ^ ks).tobytes()
                fout.write(pt)

                remaining -= n
                offset += n
                chunk_idx += 1

        mac_calc = mac.digest(length=MAC_LEN)
        if mac_calc != mac_stored:
            # Datei zurücklassen, aber warnen – je nach Policy könnte man die PT-Datei hier auch löschen.
            raise SQSEAEADError("MAC-Verify fehlgeschlagen – Cipher manipuliert oder falscher Schlüssel/Parameter.")


# ------------------- CLI -------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sqse_aead",
        description="AEAD-Verschlüsselung via SQSE (CC_OpenCL.dll): Passwort/Seed/NPY-Key, Nonce+Counter, BLAKE3-MAC."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-d", "--dll", default="CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll")
    common.add_argument("-g", "--gpu", type=int, default=0, help="GPU-Index")

    # Encrypt
    pe = sub.add_parser("encrypt", parents=[common], help="Datei verschlüsseln (AEAD)")
    pe.add_argument("infile", type=Path)
    pe.add_argument("outfile", type=Path)
    pe.add_argument("--K", type=float, default=1.2, help="chaos_K")
    pe.add_argument("--steps", type=int, default=16, help="Iterationsanzahl")
    pe.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK, help="Chunkgröße (Bytes)")
    gk = pe.add_mutually_exclusive_group(required=True)
    gk.add_argument("--pass", dest="password", type=str, help="Passwort (Argon2id)")
    gk.add_argument("--seed", type=int, help="Deterministischer Seed (für Tests)")
    gk.add_argument("--key-npy", type=Path, help="Pfad zu .npy (beliebig groß)")
    pe.add_argument("--argon-mem-mb", type=int, default=512, help="Argon2: Memory in MiB")
    pe.add_argument("--argon-t", type=int, default=3, help="Argon2: time_cost")
    pe.add_argument("--argon-p", type=int, default=4, help="Argon2: parallelism")

    # Decrypt
    pd = sub.add_parser("decrypt", parents=[common], help="Datei entschlüsseln (AEAD prüft MAC)")
    pd.add_argument("infile", type=Path)
    pd.add_argument("outfile", type=Path)
    gd = pd.add_mutually_exclusive_group(required=True)
    gd.add_argument("--pass", dest="password", type=str, help="Passwort (Argon2id)")
    gd.add_argument("--seed", type=int, help="Deterministischer Seed")
    gd.add_argument("--key-npy", type=Path, help="Pfad zu .npy")

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    try:
        if args.cmd == "encrypt":
            encrypt_file(args.infile, args.outfile, args.dll, args.gpu,
                         args.K, args.steps,
                         args.password, args.seed, args.key_npy,
                         args.chunk_bytes, args.argon_mem_mb, args.argon_t, args.argon_p)
            print(f"[ok] Encrypted: {args.infile} -> {args.outfile}")
        elif args.cmd == "decrypt":
            decrypt_file(args.infile, args.outfile, args.dll, args.gpu,
                         args.password, args.seed, args.key_npy)
            print(f"[ok] Decrypted: {args.infile} -> {args.outfile}")
        else:
            p.error("Unbekannter Befehl.")
    except SQSEAEADError as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
