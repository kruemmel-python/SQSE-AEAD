
---

# 2) Streamlit App: `app_streamlit_aead.py`

Diese App nutzt `sqse_aead.py` als Modul-Backend (gleiche Verzeichnisstruktur). Wenn du die App im selben Ordner laufen lässt, importiert sie die Funktionen aus `sqse_aead.py`. Falls du `sqse_aead.py` als Skript ohne `import` Struktur hast, nimm stattdessen die relevante Logik in die App — hier gehe ich davon aus, dass `sqse_aead.py` im selben Ordner liegt und importierbar ist.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_streamlit_aead.py

Streamlit UI für sqse_aead.py
- Key-Auswahl: Passwort / Seed / Key-npy
- DLL-Pfad (Standard: CC_OpenCL.dll) & GPU-Index
- K / steps / chunk-size
- Encrypt / Decrypt Buttons
- Anzeige der Logs & Fortschritt
"""

from __future__ import annotations
import os
import tempfile
import streamlit as st
from pathlib import Path
import subprocess
import sys
import shutil

# Wir importieren sqse_aead Funktionen (wenn sqse_aead als Modul existiert).
# Falls nicht importierbar, nutzen wir subprocess-Aufrufe.
try:
    import sqse_aead as backend
    BACKEND_IMPORTED = True
except Exception:
    BACKEND_IMPORTED = False

st.set_page_config(page_title="SQSE AEAD", layout="wide")

st.title("SQSE AEAD — GUI")

with st.sidebar:
    st.header("Allgemein")
    dll_path = st.text_input("Pfad zur CC_OpenCL.dll", "CC_OpenCL.dll")
    gpu_index = st.number_input("GPU-Index", min_value=0, value=0, step=1)
    K = st.number_input("chaos_K", min_value=0.01, value=1.2, step=0.1)
    steps = st.number_input("steps (Iterationen)", min_value=1, value=16, step=1)
    chunk_mb = st.number_input("Chunk-Größe (MiB)", min_value=1, value=8, step=1)
    chunk_bytes = int(chunk_mb * 1024 * 1024)

st.write("### Key / Auth")
key_mode = st.radio("Key-Quelle", ("Passwort (Argon2)", "Seed (deterministisch)", "Key .npy Datei"))
password = None
seed = None
key_npy = None

if key_mode == "Passwort (Argon2)":
    password = st.text_input("Passwort", type="password")
elif key_mode == "Seed (deterministisch)":
    seed = st.number_input("Seed (Ganzzahl)", min_value=0, value=123456, step=1)
else:
    key_npy_file = st.file_uploader("Key .npy Datei", type=["npy"])
    if key_npy_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        tmp.write(key_npy_file.read())
        tmp.flush()
        key_npy = Path(tmp.name)

st.write("---")

col1, col2 = st.columns(2)
with col1:
    infile = st.file_uploader("Zu verschlüsselnde/entschlüsselnde Datei", accept_multiple_files=False)
    outname = st.text_input("Zieldateiname (Relativ)", "out.aead")
with col2:
    mode = st.selectbox("Operation", ("encrypt", "decrypt"))
    submit = st.button("Ausführen")

log_area = st.empty()

def run_via_subprocess(args: list[str]):
    """Führt extern sqse_aead.py als Subprozess aus und streamt stdout/stderr."""
    cmd = [sys.executable, str(Path(__file__).parent / "sqse_aead.py")] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if line:
            out_lines.append(line)
            log_area.text("\n".join(out_lines[-200:]))
    rc = proc.wait()
    return rc, "".join(out_lines)

def copy_uploaded_to_temp(uploaded) -> Path:
    t = Path(tempfile.gettempdir()) / f"sqse_uploaded_{uploaded.name}"
    with open(t, "wb") as fh:
        fh.write(uploaded.read())
    return t

if submit:
    # Validierung
    if infile is None:
        st.error("Bitte eine Datei auswählen.")
    else:
        tmp_in = copy_uploaded_to_temp(infile)
        tmp_out = Path(tempfile.gettempdir()) / outname

        # Build CLI args
        args = [mode, str(tmp_in), str(tmp_out), "--dll", dll_path, "--gpu", str(gpu_index), "--K", str(K), "--steps", str(steps), "--chunk-bytes", str(chunk_bytes)]
        if key_mode == "Passwort (Argon2)":
            if not password:
                st.error("Bitte Passwort eingeben.")
                st.stop()
            args += ["--pass", password]
        elif key_mode == "Seed (deterministisch)":
            args += ["--seed", str(seed)]
        else:
            if key_npy is None:
                st.error("Bitte eine .npy Datei hochladen.")
                st.stop()
            args += ["--key-npy", str(key_npy)]

        st.info("Starte Verschlüsselung/Entschlüsselung... (DLL wird initialisiert; Log erscheint unten)")

        if BACKEND_IMPORTED:
            # Wenn sqse_aead importierbar ist, rufen wir die Funktionen direkt (schneller, kein Prozessstart)
            try:
                if mode == "encrypt":
                    backend.encrypt_file(Path(tmp_in), Path(tmp_out), dll_path, int(gpu_index), float(K), int(steps),
                                         password, (int(seed) if seed is not None else None), (Path(key_npy) if key_npy is not None else None),
                                         chunk_bytes, 512, 3, 4)
                else:
                    backend.decrypt_file(Path(tmp_in), Path(tmp_out), dll_path, int(gpu_index), password, (int(seed) if seed is not None else None), (Path(key_npy) if key_npy is not None else None))
                st.success("Fertig.")
                with open(tmp_out, "rb") as fh:
                    st.download_button("Download Ergebnis", fh.read(), file_name=outname)
            except Exception as e:
                st.exception(e)
        else:
            # Fallback: Subprocess
            rc, out = run_via_subprocess(args)
            if rc == 0:
                st.success("Fertig (Subprozess).")
                with open(tmp_out, "rb") as fh:
                    st.download_button("Download Ergebnis", fh.read(), file_name=outname)
            else:
                st.error(f"Fehler (RC={rc}). Log siehe oben.")
