
# SQSE AEAD — Anleitung, Best Practices & Forschungshintergrund (Windows / PowerShell)
<img width="1024" height="1024" alt="Generated Image November 03, 2025 - 7_41PM" src="https://github.com/user-attachments/assets/48f3b75e-6d73-441a-89f7-d770a5952ae7" />

Dieses Repository bündelt Demo- und Produktivwerkzeuge für **SQSE-basierte Verschlüsselung**  
(*Sub‑Quantum State Encryption*), die eine GPU‑beschleunigte DLL (`CC_OpenCL.dll`) als Keystream‑Generator nutzt.  
Das Ziel: **starke praktische Sicherheit**, klare Bedienung, reproduzierbare Workflows und saubere Integritätsprüfung (AEAD).

---

## Inhaltsverzeichnis

1. [Überblick & Modi](#überblick--modi)  
2. [Dateien & Struktur](#dateien--struktur)  
3. [Installation](#installation)  
4. [Key-Erzeugung (Passwort/Seed/NPY)](#key-erzeugung-passwortseednpy)  
5. [AEAD: Beispiele (PowerShell)](#aead-beispiele-powershell)  
6. [XOR-Modus (Legacy / Debug)](#xor-modus-legacy--debug)  
7. [Streamlit UI](#streamlit-ui)  
8. [Sicherheitshinweise](#sicherheitshinweise)  
9. [Troubleshooting](#troubleshooting)  
10. [Continuous Integration (GitHub Actions)](#continuous-integration-github-actions)  
11. [Erweiterte Werkzeuge](#erweiterte-werkzeuge)  
12. [Angriffsdauer & Kosten (Überblick)](#angriffsdauer--kosten-überblick)  
13. [Roadmap](#roadmap)  
14. [Forschung & Kryptographie‑Hintergrund (Anhang)](#forschung--kryptographiehintergrund-anhang)

---

## Überblick & Modi

Die Implementierung stellt drei Pfade bereit:

- **Legacy‑Container** (float‑basiert, *nicht* bitgenau) — rein didaktisch.  
- **XOR‑Keystream** (bitgenau) — deterministischer Keystream aus SQSE, XOR mit Plaintext.  
- **AEAD** (empfohlen) — Chunked XOR‑Keystream + **Nonce/Counter** + **Argon2id** (optional) oder Seed/NPY → **BLAKE3‑MAC**.
  - **Integrität**: Manipulationen werden erkannt (MAC).  
  - **Wiederverwendungsschutz**: Nonce/Counter pro Datei/Chunk.  
  - **Starke Schlüsselableitung** (falls Passwort): Argon2id (memory‑hard).

> ⚠️ **Hinweis:** Forschungs‑Code. Kein formaler Post‑Quantum‑Beweis.  
> Praktisch jedoch **sehr robust**, sofern Key‑Disziplin, starke KDF‑Parameter und unveränderte DLL gewährleistet sind.

---

## Dateien & Struktur

- `sqse_aead_pkg/`  
  - `core.py` — AEAD‑Kern (Header‑v2 inkl. Argon2‑Parametern, KDFs, Keystream, MAC, Selftest)  
  - `cli.py` — Typer‑CLI (encrypt / decrypt / selftest-cmd)
- `sqse_aead.py` — Standalone‑AEAD (Kompatibilitäts‑Entry)
- `sqse_files.py` — Legacy/XOR‑Modus (Debug / Demo)
- `sqse_demo.py` — Demo/Bench (quick / sweep)
- `app_streamlit_aead.py` — Streamlit GUI
- `keygen.py` — `.npy`‑Key Generator
- `CC_OpenCL.dll` — GPU‑beschleunigte Keystream‑DLL (lokal einbinden)
- **Erweiterte Tools:**  
  - `validate_header.py` — prüft Header‑v2 (Argon2‑Minima erzwingen)  
  - `rotate_keys_and_reencrypt.py` — Massen‑Rotation (parallel, atomar, verifiziert)  
  - `HSM_README.md` — Muster für HSM‑Integration (YubiHSM2, PKCS#11, CloudHSM)  
  - `cost_estimate.md` — grobe Kosten-/Zeitmodelle für Angreifer
- `.github/workflows/ci.yml` — GitHub Actions (Selftest + Header‑Check)

---

## Installation

```powershell
python -m pip install --upgrade pip
python -m pip install numpy blake3 typer rich streamlit
# Passwortmodus (Argon2id):
python -m pip install argon2-cffi
# Paket lokal (Typer-CLI):
python -m pip install -e .
```

**CLI (nach Installation):**
```powershell
sqse-aead encrypt .\in.txt .\out.aead --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --pass "SehrStarkesPasswort"
sqse-aead decrypt .\out.aead .\out.txt    --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --pass "SehrStarkesPasswort"
sqse-aead selftest-cmd --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0
```

---

## Key‑Erzeugung (Passwort/Seed/NPY)

### A) Passwort (Argon2id)
Standard (v2): `memory=512 MiB`, `t=3`, `p=4` — im Header gespeichert und beim Decrypt wiederverwendet.

### B) Seed (deterministisch)
Bequem für Tests/Reproduktion:
```powershell
--seed 123456
```

### C) `.npy` (Rohkey)
Erzeuge z. B. 64 MiB zufällige Bytes:
```powershell
python -c "import numpy as np; np.save('key.npy', np.random.default_rng(42).integers(0,256,size=64*1024*1024,dtype=np.uint8)); print('key.npy erstellt')"
# oder
python .\keygen.py 64 key.npy
```

---

## AEAD: Beispiele (PowerShell)

### Passwort / Argon2
```powershell
python .\sqse_aead.py encrypt .\test.txt .\test.aead `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 `
  --K 1.2 --steps 16 `
  --pass "SehrStarkesPasswort" `
  --argon-mem-mb 512 --argon-t 3 --argon-p 4

python .\sqse_aead.py decrypt .\test.aead .\test_out.txt `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 `
  --pass "SehrStarkesPasswort"
```

### Seed
```powershell
python .\sqse_aead.py encrypt .\test.txt .\test_seed.aead `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --seed 123456
python .\sqse_aead.py decrypt .\test_seed.aead .\test_seed_out.txt `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --seed 123456
```

### Key‑NPY
```powershell
python .\keygen.py 64 .\key.npy
python .\sqse_aead.py encrypt .\test.txt .\test_npy.aead `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --key-npy .\key.npy
python .\sqse_aead.py decrypt .\test_npy.aead .\test_npy_out.txt `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --key-npy .\key.npy
```

### Hash‑Vergleich
```powershell
Get-FileHash .\test.txt; Get-FileHash .\test_out.txt
```

---

## XOR‑Modus (Legacy / Debug)

```powershell
python .\sqse_files.py encrypt .\test.txt .\test_xor.sqse `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 `
  --K 1.2 --steps 16 --key-seed 1234 --xor

python .\sqse_files.py decrypt .\test_xor.sqse .\test_xor_out.txt `
  --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --key-seed 1234 --xor
```

---

## Streamlit UI

```powershell
streamlit run app_streamlit_aead.py
```
Funktionen: Datei‑Picker, Key‑Optionen (Pass/Seed/NPY), GPU‑Index, Chunk‑Größe, Live‑Log, Download.

---

## Sicherheitshinweise

1. **Keystream‑Reuse vermeiden** — neues `(Key, Nonce, Counter)` pro Datei/Chunk.  
2. **Argon2‑Parameter hoch genug wählen** — besonders im Passwortmodus.  
3. **MAC zwingend prüfen** — bei Fehler Datei verwerfen.  
4. **DLL schützen** — Signaturen, Integritäts‑Checks, kontrollierte Auslieferung.  
5. **Starke Schlüssel** — 256‑Bit / große `.npy` / HSM.  
6. **Kein formaler PQ‑Beweis** — praktisch robust, aber bleibt Forschungscode.

---

## Troubleshooting

- `Could not find module 'CC_OpenCL.dll'` → Pfad korrekt angeben (`--dll`), Datei ins Skript‑Verzeichnis legen.  
- `initialize_gpu RC=...` → einige Builds nutzen `1` als OK (die Tools akzeptieren das).  
- Persistente Fehler → Konsolen‑Log inkl. Kernel‑Compile‑Ausgaben posten.

---

## Continuous Integration (GitHub Actions)

- Workflow: `.github/workflows/ci.yml`  
- Enthält: Paket‑Install, optional Tests, Selftest (1 MB).  
- Da die echte DLL nicht im Repo liegt, wird im Linux‑Run ein **Mock‑Stub** angelegt. Für **Windows‑Runner**: vor dem Selftest eine signierte Test‑DLL aus geschütztem Artifact‑Store laden.

```yaml
- name: Download test DLL (Windows)
  if: runner.os == 'Windows'
  run: |
    powershell -Command "Invoke-WebRequest -Uri 'https://example.com/your-test-CC_OpenCL.dll' -OutFile 'CC_OpenCL.dll'"
```

**Wichtig:** Keine produktive DLL in öffentlichen CI‑Runs veröffentlichen.

---

## Erweiterte Werkzeuge

### `validate_header.py` — Argon2‑Minima erzwingen
- Liest **Header‑v2** und prüft `argon_m/t/p` gegen Mindestwerte (per ENV überschreibbar).
- Einsatz: Release‑QA, Ingest‑Pipelines, CI‑Checks.

### `rotate_keys_and_reencrypt.py` — Massen‑Rotation
- Parallel, atomar (Backup → Replace), **Roundtrip‑Verifikation** jeder Datei.  
- Für Millionen Dateien: Chunk‑Weise, Logging, Wiederaufnahmepunkte ergänzen.

### `HSM_README.md` — HSM‑Integration
- Muster/Pseudocode für YubiHSM2 / PKCS#11 / CloudHSM.  
- Empfehlung: Masterkey **nie exportieren**, Datei‑Keys kontextuell ableiten (`HKDF(master, nonce||fileid)`).

### `cost_estimate.md` — Kostenmodelle
- Grobe Rechnungen: warum Argon2 + starke Keys realistische Angreifer ausbremst.  
- **Kernaussage:** Brute‑Force wird ökonomisch schnell absurd teuer.

---

## Angriffsdauer & Kosten (Überblick)

- **128‑Bit und mehr** → Brute‑Force praktisch unmöglich (astronomische Zeiten).  
- **Passwörter**: tatsächliche Angriffe zielen auf Wörterbuch + Regeln + GPU — Argon2 (z. B. 512 MiB, t=3) reduziert Versuchsrate massiv.  
- **Risikotreiber**: gestohlene `.npy`, kompromittierte DLL/Host, Side‑Channels. Schütze Schlüsselmaterial & Infrastruktur.

---

## Roadmap

- HSM‑Schlüsselquellen „first‑class“ (PKCS#11‑Backend)  
- Header‑Metadaten: Dateiname, Zeitstempel, Kommentar (optional)  
- Mehr‑GPU Parallelisierung & Benchmarks (MB/s/Chunkgröße)  
- UI‑Reports (Selftest/Throughput/Argon2‑Timing)  
- Optionale „deterministic counter“‑Policy pro Datei

---

## Forschung & Kryptographie‑Hintergrund (Anhang)

> Dieser Abschnitt erklärt die Idee hinter **SQSE** (Sub‑Quantum State Encryption) und ihre sichere Einbettung als Keystream‑Quelle in ein klassisches AEAD‑Gerüst.

### 1) Motivation
- **Klassische Stromchiffren** (CTR/XChaCha20/Etc.) benötigen einen starken Pseudozufall‑Keystream.  
- **SQSE** liefert einen **GPU‑beschleunigten, komplexen Keystream** aus gekoppelten **dynamischen Systemen** (chaotische Karten, feldbasierte Iterationen, interne Rückkopplungen).  
- Die **Sicherheit** im praktischen Betrieb entsteht aus *zwei* Schichten:  
  1) **Starkes Schlüsselmaterial** (Argon2/Seed/NPY/HSM)  
  2) **Korrekte AEAD‑Einbettung** (Nonce/Counter, MAC, Keystream‑Disziplin)

### 2) Dynamische Kerne (vereinfacht)
- **Chirikov‑/Standard‑Map**‑artige Iteration mit Parameter **K** (Chaos‑Intensität), Zustandsvektoren (θ, p), Iterationszahl **T**.  
- **Kopplungen/Masken** (z. B. `p_mask`) stabilisieren/verrauschen lokale Zustände; GPU‑Kerne mischen Zustandsräume in hoher Dimension.
- Ergebnis: **hochgradig sensitiv** gegenüber Schlüssel/Seed/Parametern → praktisch nicht invertierbar ohne Geheimnis.

> Wichtig: Die **Korrektheit** der Verschlüsselung hängt *nicht* allein an der „Unknackbarkeit“ der Dynamik,  
> sondern an der **AEAD‑Konstruktion** (Nonce/Counter + MAC). Darum ist die Keystream‑Quelle modular austauschbar — SQSE ist eine performante Option.

### 3) AEAD‑Design (praktisch)
- Datei wird in **Chunks** gesplittet (z. B. 8 MiB).  
- Für jeden Chunk: deterministische **Keystream‑Ableitung** aus (Key, Nonce, Counter[, K, T]).  
- **XOR** mit Plaintext → Ciphertext; zusätzlich **BLAKE3‑MAC** auf `(Header || Nonce || Counter || CipherChunk)` → Integrität.  
- **Header‑v2** speichert `argon_m/t/p` und technische Parameter → reproduzierbare Entschlüsselung.

### 4) Threat Model & Implikationen
- **Brute‑Force** gegen 256‑Bit Key ist irrelevant (astronomisch).  
- **Password‑Guessing** ist realistisch → **Argon2id** macht jeden Versuch teuer (Memory‑Hard).  
- **Keystream‑Reuse** wäre fatal (XOR‑Chiffren‑Klasse) → korrekte Nonce/Counter‑Politik ist entscheidend (erfüllt).  
- **Implementation Risks**: kompromittierte DLL/Host, Side‑Channels → Integritätsprüfung, Signaturen, Härtung.

### 5) Parameterwahl
- **K (Chaos‑Parameter)**: Mittelwerte (z. B. 1.2–2.0) liefern gute Mischung ohne Divergenz; höhere K erhöhen Entropietransfer, aber teste Stabilität.  
- **steps (T)**: erhöht Mischungsgrad; wachse moderat (16–64), beachte Performance vs. Nutzen.  
- **argon_m/t/p**: an Bedrohungslage anpassen (Laptop 64–128 MiB; Enterprise 256–512 MiB; High‑Threat ≥ 1024 MiB).

### 6) Verifikation & Bench
- **Selftest** prüft Roundtrip + MAC.  
- **Benchmarks**: Messe Durchsatz (MB/s) vs. Chunkgröße; prüfe GPU‑Sättigung (DLL‑Kompatibilität/Kernels).

### 7) Reproduzierbarkeit & Audit
- **Header‑v2** + `validate_header.py` → Policy‑konforme Entschlüsselbarkeit.  
- **rotate_keys_and_reencrypt.py** → sichere Migration/Rotation mit Verifizierung.  
- **CI‑Checks** → Mindest‑Argon2 erzwingen, Selftest dokumentieren.

---

## Kontakt / Credits

**Entwickler:** Ralf Krümmel  
**Projekt:** CC_OpenCL / SQSE (Sub‑Quantum State Encryption)  

