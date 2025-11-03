#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_streamlit.py
================

Interaktive SQSE-Visualisierung gegen CC_OpenCL.dll:
- GPU-Index auswählbar
- Kernel auto-laden (embedded)
- Encrypt+Decrypt Roundtrip mit MAE
- Zwei Plots: (1) Original vs. Rekonstruktion (erste N Punkte), (2) Histogramm θ_T

Chart-Regeln: matplotlib, 1 Plot pro Figur, keine Farb-Styles erzwingen.
"""

from __future__ import annotations
import os
from ctypes import cdll, c_int, c_float, c_char_p, POINTER
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------- ctypes Helper -----------------------------------
def as_ptr(a: np.ndarray):
    return a.ctypes.data_as(POINTER(c_float))

def ensure_f32_c(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a

def resolve_dll_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, path)

# --------------------------- DLL Laden (Cache) -------------------------------
@st.cache_resource
def load_lib(dll_path: str, gpu_index: int):
    dll_path = resolve_dll_path(dll_path)
    dll = cdll.LoadLibrary(dll_path)

    # initialize_gpu: Erfolg==1
    if hasattr(dll, "initialize_gpu"):
        dll.initialize_gpu.argtypes = [c_int]
        dll.initialize_gpu.restype  = c_int
        rc = dll.initialize_gpu(gpu_index)
        if rc != 1:
            st.error(f"initialize_gpu({gpu_index}) RC={rc} (erwartet 1)")
            st.stop()

    # SQSE binden: Erfolg==0
    dll.sqse_load_kernels.argtypes = [c_char_p]
    dll.sqse_load_kernels.restype  = c_int
    rc = dll.sqse_load_kernels(b"embedded")
    if rc != 0:
        st.error(f"sqse_load_kernels RC={rc}")
        st.stop()

    dll.execute_sqse_encrypt_float.argtypes = [
        POINTER(c_float), POINTER(c_float),
        c_int, c_float, c_int,
        POINTER(c_float), POINTER(c_float),
    ]
    dll.execute_sqse_encrypt_float.restype = c_int

    dll.execute_sqse_decrypt_float.argtypes = [
        POINTER(c_float), POINTER(c_float), POINTER(c_float),
        c_int, c_float, c_int,
        POINTER(c_float),
    ]
    dll.execute_sqse_decrypt_float.restype = c_int

    return dll

# --------------------------- App --------------------------------------------
def run():
    st.title("SQSE – Sub-Quantum State Encryption (Demo)")

    col1, col2 = st.columns(2)
    with col1:
        dll_path = st.text_input("Pfad zur CC_OpenCL.dll", "CC_OpenCL.dll")
        gpu = st.number_input("GPU-Index", min_value=0, value=0, step=1)
        seed = st.number_input("Zufalls-Seed", min_value=0, value=42, step=1)
    with col2:
        n = st.slider("Länge N", 128, 8192, 1024, step=128)
        K = st.slider("chaos_K (Standard-Map K)", 0.2, 6.0, 1.2, step=0.1)
        steps = st.slider("Iterationen", 1, 128, 16)

    dll = load_lib(dll_path, int(gpu))

    rng = np.random.default_rng(int(seed))
    data = ensure_f32_c(rng.random(int(n), dtype=np.float32))
    key  = ensure_f32_c(rng.random(int(n), dtype=np.float32))

    theta = np.empty_like(data)
    pmask = np.empty_like(data)
    rec   = np.empty_like(data)

    if st.button("Encrypt + Decrypt"):
        rc = dll.execute_sqse_encrypt_float(as_ptr(data), as_ptr(key), c_int(int(n)),
                                            c_float(float(K)), c_int(int(steps)),
                                            as_ptr(theta), as_ptr(pmask))
        st.write(f"Encrypt RC={rc}")
        if rc == 0:
            rc = dll.execute_sqse_decrypt_float(as_ptr(theta), as_ptr(pmask), as_ptr(key), c_int(int(n)),
                                                c_float(float(K)), c_int(int(steps)),
                                                as_ptr(rec))
            st.write(f"Decrypt RC={rc}")

        if rc == 0:
            mae = float(np.mean(np.abs(data - rec)))
            st.metric("Roundtrip MAE", f"{mae:.3e}")

            # Plot 1: Original vs Rekonstruktion
            idxN = min(256, int(n))
            fig1, ax1 = plt.subplots()
            ax1.plot(np.arange(idxN), data[:idxN], label="original")
            ax1.plot(np.arange(idxN), rec[:idxN], label="reconstruct", linestyle="--")
            ax1.set_title("Original vs. Rekonstruktion (erste 256)")
            ax1.legend()
            st.pyplot(fig1)

            # Plot 2: Histogramm von θ_T
            fig2, ax2 = plt.subplots()
            ax2.hist(theta, bins=50)
            ax2.set_title("Verteilung θ_T")
            st.pyplot(fig2)

if __name__ == "__main__":
    run()
