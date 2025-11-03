
from __future__ import annotations
import os, struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from ctypes import cdll, c_int, c_float, c_char_p, POINTER

try:
    from blake3 import blake3
except Exception as e:
    raise RuntimeError("Bitte 'pip install blake3' installieren.") from e

try:
    from argon2.low_level import hash_secret_raw, Type as Argon2Type
    ARGON2_AVAILABLE = True
except Exception:
    ARGON2_AVAILABLE = False

# --------- Format & Konstanten ----------
MAGIC = b"SX3A"
VERSION = 2  # v2: Argon2-Parameter (m,t,p) im Header

FLAG_AEAD   = 1 << 0
FLAG_ARGON2 = 1 << 1

MAC_LEN   = 32
NONCE_LEN = 16
SALT_LEN  = 16

DEFAULT_CHUNK = 8 * 1024 * 1024

class SQSEAEADError(Exception): ...

def as_ptr(a: np.ndarray):
    return a.ctypes.data_as(POINTER(c_float))

def ensure_f32_c(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a

class SQSEDLL:
    def __init__(self, path: str, gpu_index: int):
        path = self._resolve_path(path)
        try:
            self.lib = cdll.LoadLibrary(path)
        except OSError as e:
            raise SQSEAEADError(f"DLL konnte nicht geladen werden: {e}")
        if hasattr(self.lib, "initialize_gpu"):
            self.lib.initialize_gpu.argtypes = [c_int]
            self.lib.initialize_gpu.restype  = c_int
            rc = self.lib.initialize_gpu(gpu_index)
            if rc != 1:
                raise SQSEAEADError(f"initialize_gpu({gpu_index}) RC={rc} (erwartet 1)")
        self.lib.sqse_load_kernels.argtypes = [c_char_p]
        self.lib.sqse_load_kernels.restype  = c_int
        rc = self.lib.sqse_load_kernels(b"embedded")
        if rc != 0:
            raise SQSEAEADError(f"sqse_load_kernels RC={rc}")
        self.lib.execute_sqse_encrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int, POINTER(c_float), POINTER(c_float)
        ]
        self.lib.execute_sqse_encrypt_float.restype  = c_int
        self.lib.execute_sqse_decrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int, POINTER(c_float)
        ]
        self.lib.execute_sqse_decrypt_float.restype  = c_int

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isabs(path): return path
        from os.path import dirname, abspath, join
        return join(dirname(abspath(__file__)), "..", path)

    def encrypt_vec(self, x: np.ndarray, key: np.ndarray, K: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
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

# ---------- KDFs ----------
def kdf_from_password(password: str, salt: bytes, mem_mb: int, time_cost: int, parallelism: int) -> bytes:
    if not ARGON2_AVAILABLE:
        raise SQSEAEADError("Argon2 nicht verfügbar. Bitte 'pip install argon2-cffi'.")
    mem_kib = mem_mb * 1024
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=mem_kib,
        parallelism=parallelism,
        hash_len=32,
        type=Argon2Type.ID,
    )

def kdf_from_seed(seed: int) -> bytes:
    h = blake3()
    h.update(b"SQSE-SEED-KEY\0")
    h.update(seed.to_bytes(16, "little", signed=False))
    return h.digest(length=32)

def kdf_from_npy(npy_path: Path) -> bytes:
    arr = np.load(str(npy_path))
    blob = memoryview(np.ascontiguousarray(arr)).tobytes()
    return blake3(blob).digest(length=32)

def expand_key_float32(n: int, ctx_key: bytes, nonce: bytes, counter: int) -> np.ndarray:
    need = n * 4
    hk = blake3(key=ctx_key)
    hk.update(b"SQSE-F32-EXPAND\0")
    hk.update(nonce)
    hk.update(counter.to_bytes(8, "little"))
    stream = hk.digest(length=need)
    u32 = np.frombuffer(stream, dtype=np.uint32, count=n)
    return (u32.astype(np.float64) / (2**32)).astype(np.float32)

# ---------- Keystream ----------
def keystream_bytes(dll: SQSEDLL, n: int, chaos_K: float, steps: int, key_f32: np.ndarray) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    theta, _ = dll.encrypt_vec(x, key_f32, chaos_K, steps)
    frac = theta - np.floor(theta)
    return np.floor(frac * 256.0).astype(np.uint8)

# ---------- Header mit Argon2-Parametern (v2) ----------
# <4sBBH f I I Q H H I I I
# MAGIC, VER, FLAGS, reserved, K, steps, chunk_bytes, total_bytes, salt_len, nonce_len, argon_m, argon_t, argon_p
def write_header_v2(fh, flags: int, K: float, steps: int, chunk_bytes: int,
                    total_bytes: int, salt: bytes, nonce: bytes,
                    argon_m: int, argon_t: int, argon_p: int) -> None:
    hdr = struct.pack("<4sBBHfIIQHHIII",
                      MAGIC, VERSION, flags, 0,
                      np.float32(K).item(), int(steps), int(chunk_bytes),
                      int(total_bytes), len(salt), len(nonce),
                      int(argon_m), int(argon_t), int(argon_p))
    fh.write(hdr); fh.write(salt); fh.write(nonce)

def read_header_v2(fh) -> tuple[int, float, int, int, int, bytes, bytes, int, int, int]:
    base_len = struct.calcsize("<4sBBHfIIQHHIII")
    base = fh.read(base_len)
    if len(base) < base_len:
        raise SQSEAEADError("Header zu kurz.")
    magic, ver, flags, _res, K, steps, chunk_bytes, total_bytes, slen, nlen, amem, atime, apar = \
        struct.unpack("<4sBBHfIIQHHIII", base)
    if magic != MAGIC or ver != VERSION:
        raise SQSEAEADError("Unbekanntes Format oder Version.")
    salt = fh.read(slen); nonce = fh.read(nlen)
    if len(salt) != slen or len(nonce) != nlen:
        raise SQSEAEADError("Header Felder unvollständig.")
    return flags, float(K), int(steps), int(chunk_bytes), int(total_bytes), salt, nonce, int(amem), int(atime), int(apar)

def mac_update_v2(h, flags, K, steps, chunk_bytes, total_bytes, salt, nonce, amem, atime, apar):
    h.update(MAGIC); h.update(bytes([VERSION])); h.update(bytes([flags])); h.update(b"\x00\x00")
    h.update(np.float32(K).tobytes())
    import struct as _s
    h.update(_s.pack("<I", steps)); h.update(_s.pack("<I", chunk_bytes)); h.update(_s.pack("<Q", total_bytes))
    h.update(_s.pack("<H", len(salt))); h.update(_s.pack("<H", len(nonce)))
    h.update(_s.pack("<I", amem)); h.update(_s.pack("<I", atime)); h.update(_s.pack("<I", apar))
    h.update(salt); h.update(nonce)

# ---------- Encrypt / Decrypt ----------
def encrypt_file(in_path: Path, out_path: Path, dll_path: str, gpu: int,
                 K: float, steps: int,
                 password: Optional[str], seed: Optional[int], key_npy: Optional[Path],
                 chunk_bytes: int, argon_mem_mb: int, argon_t: int, argon_p: int) -> None:
    pt_size = in_path.stat().st_size
    salt  = os.urandom(SALT_LEN) if password else (b"" if (seed is not None or key_npy) else os.urandom(SALT_LEN))
    nonce = os.urandom(NONCE_LEN)
    if password:
        key32 = kdf_from_password(password, salt, argon_mem_mb, argon_t, argon_p)
        flags = FLAG_AEAD | FLAG_ARGON2
        amem, atime, apar = argon_mem_mb, argon_t, argon_p
    elif seed is not None:
        key32 = kdf_from_seed(int(seed)); flags = FLAG_AEAD; amem=atime=apar=0
    elif key_npy is not None:
        key32 = kdf_from_npy(key_npy); flags = FLAG_AEAD; amem=atime=apar=0
    else:
        raise SQSEAEADError("Bitte --pass oder --seed oder --key-npy angeben.")
    dll = SQSEDLL(dll_path, gpu)
    from blake3 import blake3 as _b3
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        write_header_v2(fout, flags, K, steps, chunk_bytes, pt_size, salt, nonce, amem, atime, apar)
        mac = _b3(key=key32)
        mac_update_v2(mac, flags, K, steps, chunk_bytes, pt_size, salt, nonce, amem, atime, apar)
        remaining = pt_size; chunk_idx = 0
        while remaining > 0:
            n = min(chunk_bytes, remaining)
            block = fin.read(n)
            if len(block) != n: raise SQSEAEADError("Kurzlese im Plaintext.")
            key_f32 = expand_key_float32(n, key32, nonce, chunk_idx)
            ks = keystream_bytes(dll, n, K, steps, key_f32)
            ct = (np.frombuffer(block, dtype=np.uint8) ^ ks).tobytes()
            fout.write(ct); mac.update(ct)
            remaining -= n; chunk_idx += 1
        fout.write(mac.digest(length=MAC_LEN))

def decrypt_file(in_path: Path, out_path: Path, dll_path: str, gpu: int,
                 password: Optional[str], seed: Optional[int], key_npy: Optional[Path]) -> None:
    with open(in_path, "rb") as fin:
        flags, K, steps, chunk_bytes, total_bytes, salt, nonce, amem, atime, apar = read_header_v2(fin)
        if flags & FLAG_ARGON2:
            if not password: raise SQSEAEADError("Datei erwartet Passwort/Argon2, aber --pass fehlt.")
            if not ARGON2_AVAILABLE: raise SQSEAEADError("Argon2 nicht verfügbar. Bitte 'pip install argon2-cffi'.")
            key32 = kdf_from_password(password, salt, amem, atime, apar)  # Nutze Header-Parameter!
        else:
            if seed is not None: key32 = kdf_from_seed(int(seed))
            elif key_npy is not None: key32 = kdf_from_npy(key_npy)
            else: raise SQSEAEADError("Datei erwartet Seed/NPY, Parameter fehlen.")
        dll = SQSEDLL(dll_path, gpu)
        rest = fin.read()
        if len(rest) < total_bytes + MAC_LEN: raise SQSEAEADError("Cipher/MAC unvollständig.")
        cipher = rest[:total_bytes]; mac_stored = rest[total_bytes: total_bytes + MAC_LEN]
        from blake3 import blake3 as _b3
        mac = _b3(key=key32); mac_update_v2(mac, flags, K, steps, chunk_bytes, total_bytes, salt, nonce, amem, atime, apar)
        with open(out_path, "wb") as fout:
            remaining = total_bytes; chunk_idx = 0; offset = 0
            while remaining > 0:
                n = min(chunk_bytes, remaining)
                ct = cipher[offset: offset + n]; mac.update(ct)
                key_f32 = expand_key_float32(n, key32, nonce, chunk_idx)
                ks = keystream_bytes(dll, n, K, steps, key_f32)
                pt = (np.frombuffer(ct, dtype=np.uint8) ^ ks).tobytes()
                fout.write(pt)
                remaining -= n; offset += n; chunk_idx += 1
        if mac.digest(length=MAC_LEN) != mac_stored:
            raise SQSEAEADError("MAC-Verify fehlgeschlagen – falscher Schlüssel/Parameter oder manipulierte Datei.")

# ---------- Self-Test ----------
def selftest(dll_path: str, gpu: int, K: float = 1.2, steps: int = 16, seed: int = 1234, size: int = 1_000_000) -> tuple[bool, str]:
    """Verschlüsselt/entschlüsselt 1 MB Per-Demo (Seed) im Speicher und prüft MAC/Bitgleichheit."""
    import tempfile
    pt = os.urandom(size)
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td)/"pt.bin"; enc = Path(td)/"ct.aead"; dec = Path(td)/"dec.bin"
        inp.write_bytes(pt)
        encrypt_file(inp, enc, dll_path, gpu, K, steps, None, seed, None, DEFAULT_CHUNK, 512, 3, 4)
        decrypt_file(enc, dec, dll_path, gpu, None, seed, None)
        ok = (pt == dec.read_bytes())
        return ok, f"Selftest OK: {ok}, size={size}"
