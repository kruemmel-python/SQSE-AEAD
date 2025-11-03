#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqse_demo.py
============

GPU-auswählbare Demo für die SQSE-Implementierung in CC_OpenCL.dll.

- DLL sauber via ctypes binden (exakte Signaturen)
- initialize_gpu(gpu_index): Erfolgscode deiner DLL ist 1
- sqse_load_kernels(...), execute_sqse_encrypt_float(...), execute_sqse_decrypt_float(...): Erfolgscode 0
- Roundtrip (Encrypt -> Decrypt) mit MAE-Prüfung
- Modi: quick | sweep
- Python 3.12: match/case, dataclasses(slots=True)

WICHTIG:
- SQSE arbeitet auf float32 in [0,1). Für Bytes: vorher byte/255.0, nachher *255.0 zurückquantisieren.
"""

from __future__ import annotations

from ctypes import cdll, c_int, c_float, c_char_p, POINTER
from dataclasses import dataclass
from typing import Literal, Optional
import argparse
import sys
import os
import numpy as np


# --------------------------- Fehlerklasse ------------------------------------
class SQSEError(Exception):
    """Fehlerklasse für Binding- und Laufzeitfehler in der SQSE-Demo."""


# --------------------------- Parameter-Container -----------------------------
@dataclass(slots=True)
class SQSEParams:
    chaos_K: float = 1.2
    steps: int = 16
    n: int = 1024
    seed: int = 42
    gpu_index: int = 0


# --------------------------- Array-Helfer ------------------------------------
def ensure_f32_c(arr: np.ndarray, name: str) -> np.ndarray:
    """Erzwingt float32 + C-Layout + 1D-Form für OpenCL/ctypes."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.ndim != 1:
        raise SQSEError(f"{name}: 1D-Array erwartet, erhalten shape={arr.shape}")
    return arr


def as_c_float_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(POINTER(c_float))


# --------------------------- DLL-Bindings ------------------------------------
@dataclass(slots=True)
class DLLBindings:
    lib: any
    initialize_gpu: Optional[any] = None
    sqse_load_kernels: any = None
    execute_sqse_encrypt_float: any = None
    execute_sqse_decrypt_float: any = None

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Falls der Pfad relativ ist, relativ zum Skriptverzeichnis auflösen."""
        if os.path.isabs(path):
            return path
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    @staticmethod
    def load(path: str = "CC_OpenCL.dll") -> "DLLBindings":
        path = DLLBindings._resolve_path(path)
        try:
            lib = cdll.LoadLibrary(path)
        except OSError as e:
            raise SQSEError(f"DLL konnte nicht geladen werden: {e}") from e
        return DLLBindings(lib=lib)

    def bind(self) -> None:
        """Exakte ctypes-Signaturen setzen (Type-Safety statt UB)."""
        # Optional: GPU-Init
        if hasattr(self.lib, "initialize_gpu"):
            self.lib.initialize_gpu.argtypes = [c_int]
            self.lib.initialize_gpu.restype = c_int
            self.initialize_gpu = self.lib.initialize_gpu
        else:
            self.initialize_gpu = None

        # SQSE-Kernel laden
        self.lib.sqse_load_kernels.argtypes = [c_char_p]
        self.lib.sqse_load_kernels.restype = c_int
        self.sqse_load_kernels = self.lib.sqse_load_kernels

        # Encrypt
        self.lib.execute_sqse_encrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float), POINTER(c_float),
        ]
        self.lib.execute_sqse_encrypt_float.restype = c_int
        self.execute_sqse_encrypt_float = self.lib.execute_sqse_encrypt_float

        # Decrypt
        self.lib.execute_sqse_decrypt_float.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_int, c_float, c_int,
            POINTER(c_float),
        ]
        self.lib.execute_sqse_decrypt_float.restype = c_int
        self.execute_sqse_decrypt_float = self.lib.execute_sqse_decrypt_float


# --------------------------- SQSE-Kernroutinen -------------------------------
def sqse_encrypt(bind: DLLBindings, data: np.ndarray, key: np.ndarray,
                 chaos_K: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    data = ensure_f32_c(data, "data")
    key = ensure_f32_c(key, "key")
    n = data.size
    theta = np.empty(n, dtype=np.float32)
    pmask = np.empty(n, dtype=np.float32)
    rc = bind.execute_sqse_encrypt_float(
        as_c_float_ptr(data), as_c_float_ptr(key),
        c_int(n), c_float(chaos_K), c_int(steps),
        as_c_float_ptr(theta), as_c_float_ptr(pmask),
    )
    if rc != 0:
        raise SQSEError(f"Encrypt RC={rc}")
    return theta, pmask


def sqse_decrypt(bind: DLLBindings, theta: np.ndarray, pmask: np.ndarray, key: np.ndarray,
                 chaos_K: float, steps: int) -> np.ndarray:
    theta = ensure_f32_c(theta, "theta")
    pmask = ensure_f32_c(pmask, "p_masked")
    key = ensure_f32_c(key, "key")
    if not (theta.size == pmask.size == key.size):
        raise SQSEError(f"Größeninkonsistenz: {theta.size}/{pmask.size}/{key.size}")
    n = theta.size
    rec = np.empty(n, dtype=np.float32)
    rc = bind.execute_sqse_decrypt_float(
        as_c_float_ptr(theta), as_c_float_ptr(pmask), as_c_float_ptr(key),
        c_int(n), c_float(chaos_K), c_int(steps),
        as_c_float_ptr(rec),
    )
    if rc != 0:
        raise SQSEError(f"Decrypt RC={rc}")
    return rec


def sqse_roundtrip(bind: DLLBindings, data: np.ndarray, key: np.ndarray,
                   chaos_K: float, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    theta, pmask = sqse_encrypt(bind, data, key, chaos_K, steps)
    recon = sqse_decrypt(bind, theta, pmask, key, chaos_K, steps)
    mae = float(np.mean(np.abs(recon - data)))
    return recon, theta, mae


# --------------------------- Demo-Runner -------------------------------------
def run_demo(mode: Literal["quick", "sweep"], params: SQSEParams, dll_path: str) -> None:
    bind = DLLBindings.load(dll_path)
    bind.bind()

    # initialize_gpu: Erfolg == 1
    if bind.initialize_gpu is not None:
        rc = bind.initialize_gpu(c_int(params.gpu_index))
        if rc != 1:
            raise SQSEError(f"initialize_gpu({params.gpu_index}) RC={rc} (erwartet 1)")
        print(f"[info] GPU initialisiert: index={params.gpu_index}")
    else:
        print("[warn] initialize_gpu nicht exportiert; fahre fort.")

    # Kernel laden: Erfolg == 0
    rc = bind.sqse_load_kernels(b"embedded")
    if rc != 0:
        raise SQSEError(f"sqse_load_kernels RC={rc}")

    rng = np.random.default_rng(params.seed)
    data = rng.random(params.n, dtype=np.float32)
    key = rng.random(params.n, dtype=np.float32)

    match mode:
        case "quick":
            recon, _, mae = sqse_roundtrip(bind, data, key, params.chaos_K, params.steps)
            print(f"[quick] GPU={params.gpu_index} N={params.n} "
                  f"K={params.chaos_K:.3f} steps={params.steps} -> MAE={mae:.6e}")
        case "sweep":
            Ks = (0.8, 1.2, 2.0, 4.0)
            Steps = (8, 16, 32, 64)
            print(f"[sweep] GPU={params.gpu_index} N={params.n}")
            for K in Ks:
                for s in Steps:
                    _, _, mae = sqse_roundtrip(bind, data, key, K, s)
                    print(f"  K={K:3.1f} steps={s:2d} -> MAE={mae:.6e}")
        case _:
            raise SQSEError(f"Unbekannter Modus: {mode!r}")


# --------------------------- CLI-PARSER --------------------------------------
def parse_args(argv: list[str]) -> tuple[Literal["quick", "sweep"], SQSEParams, str]:
    p = argparse.ArgumentParser(
        prog="sqse_demo",
        description="SQSE-Demo (chaos-basierte Verschlüsselung) gegen CC_OpenCL.dll – mit GPU-Auswahl."
    )
    p.add_argument("-m", "--mode", choices=["quick", "sweep"], default="quick")
    p.add_argument("-d", "--dll", default="CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--K", type=float, default=1.2)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-g", "--gpu", type=int, default=0, help="GPU-Index (initialize_gpu)")
    args = p.parse_args(argv)
    params = SQSEParams(
        chaos_K=float(args.K), steps=int(args.steps),
        n=int(args.n), seed=int(args.seed),
        gpu_index=int(args.gpu),
    )
    return args.mode, params, args.dll


def main() -> None:
    mode, params, dll_path = parse_args(sys.argv[1:])
    run_demo(mode, params, dll_path)


if __name__ == "__main__":
    main()
