# SQSE AEAD — Anleitung und Beispiele (Windows / PowerShell)

Dieses Repository enthält Demo- und Produktivwerkzeuge für **SQSE-basierte Verschlüsselung** (Sub-Quantum State Encryption), welche die GPU/DLL `CC_OpenCL.dll` als Keystream-Generator nutzt.  
Die Implementierung bietet mehrere Modi:

- **Legacy-Container** (float-basierter Container, nicht bitgenau) — nur zu Demonstrationszwecken.  
- **XOR-Keystream** (bitgenau) — erzeugt einen deterministischen Keystream aus SQSE und XORt diesen mit dem Plaintext (bitgenaue Roundtrips).  
- **AEAD (sqse_aead.py)** — Produktionstauglicher Modus: Chunked XOR-Keystream + **Nonce/Counter** + **Argon2id** (optional) oder Seed/NPY → **BLAKE3-MAC** für Integrität.

> Hinweis: Dies ist Forschungscode und kein ersetzt formell verifizierte Post-Quantum-Kryptografie. Er liefert jedoch starke praktische Schutzmechanismen (memory-hard KDF, nonce/counter, AEAD-MAC).

---

## Dateien in diesem Ordner

- `sqse_aead.py` — AEAD-Verschlüsselung (empfohlen).  
- `sqse_files.py` — ältere/leichtere Varianten (Legacy/XOR).  
- `sqse_demo.py` — Demo / Bench (quick / sweep).  
- `app_streamlit_aead.py` — Streamlit UI für `sqse_aead.py`.  
- `keygen.py` (optional) — Hilfsprogramm zum Erzeugen einer `.npy` Keydatei.  
- `CC_OpenCL.dll` — deine kompilierte DLL (muss vorhanden sein).

---

## Voraussetzungen / Installation

Python 3.12 empfohlen. Im Powershell (einzeilig):

```powershell
python -m pip install --upgrade pip
python -m pip install numpy blake3 streamlit
# Optional (für Passwortmodus):
python -m pip install argon2-cffi
````

> Wenn du keine Argon2-Abhängigkeit willst, nutze `--seed` oder `--key-npy` statt `--pass`.

---

## Key-Erzeugung

Du hast drei Möglichkeiten, ein Schlüsselmaterial bereitzustellen:

### A) Passwort (Argon2id)

Sicherste, menschenlesbare Option. Argon2-Parameter sind standardmäßig in `sqse_aead.py` gesetzt (512 MiB memory, t=3, p=4). Du musst nichts erzeugen — gib beim Encrypt/Decrypt `--pass "MeinStarkesPasswort"` an.

### B) Seed (deterministisch)

Für Tests oder reproduzierbare Laufwege:

```powershell
--seed 123456
```

Das erzeugt intern einen 32-Byte Key via BLAKE3.

### C) `.npy` Datei (Rohkey)

Große, zufällige Binärdatei als Key. Erzeuge z.B. eine 64 MiB Keydatei (PowerShell-Einzeiler):

```powershell
python -c "import numpy as np; np.save('key.npy', np.random.default_rng(42).integers(0,256,size=64*1024*1024,dtype=np.uint8)); print('key.npy erstellt')"
```

Oder benutze das mitgelieferte `keygen.py`:

```powershell
python .\keygen.py 64 key.npy
```

---

## AEAD: Beispiele (PowerShell, Einzeiler)

### Verschlüsseln (Passwort / Argon2)

```powershell
python .\sqse_aead.py encrypt .\test.txt .\test.aead --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --pass "SehrStarkesPasswort" --argon-mem-mb 512 --argon-t 3 --argon-p 4
```

### Entschlüsseln (Passwort)

```powershell
python .\sqse_aead.py decrypt .\test.aead .\test_out.txt --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --pass "SehrStarkesPasswort"
```

### Verschlüsseln (Seed)

```powershell
python .\sqse_aead.py encrypt .\test.txt .\test_seed.aead --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --seed 123456
python .\sqse_aead.py decrypt .\test_seed.aead .\test_seed_out.txt --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --seed 123456
```

### Verschlüsseln (Key aus `.npy`)

```powershell
# Keyfile erzeugen (falls noch nicht vorhanden)
python -c "import numpy as np; np.save('key.npy', np.random.default_rng(42).integers(0,256,size=64*1024*1024,dtype=np.uint8)); print('key.npy erstellt')"

# Encrypt/Decrypt mit key.npy
python .\sqse_aead.py encrypt .\test.txt .\test_npy.aead --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --key-npy .\key.npy
python .\sqse_aead.py decrypt .\test_npy.aead .\test_npy_out.txt --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --key-npy .\key.npy
```

### Hashvergleich (prüft Bitgleichheit)

```powershell
Get-FileHash .\test.txt; Get-FileHash .\test_out.txt
```

---

## `sqse_files.py` (XOR-Modus) — kurze Befehle (falls du nur XOR willst)

```powershell
# Encrypt (bitgenau)
python .\sqse_files.py encrypt .\test.txt .\test_xor.sqse --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --K 1.2 --steps 16 --key-seed 1234 --xor

# Decrypt
python .\sqse_files.py decrypt .\test_xor.sqse .\test_xor_out.txt --dll "G:\verschluesselung\CC_OpenCL.dll" --gpu 0 --key-seed 1234 --xor

# Hash-Check
Get-FileHash .\test.txt; Get-FileHash .\test_xor_out.txt
```

---

## Streamlit UI (lokal)

Starte die App im Ordner mit `app_streamlit_aead.py`:

```powershell
streamlit run app_streamlit_aead.py
```

Die App bietet:

* Dateiwähler für Input/Output
* Key-Optionen (Passwort / Seed / Key-npy)
* GPU-Index Feld
* Chunk-Größe Einstellung
* „Encrypt“ / „Decrypt“ Buttons und Live-Protokoll der DLL-Ausgaben
* Optional: Download-Link der erzeugten Datei (Streamlit stellt dies bereit)

---

## Sicherheitshinweise (essentiell)

1. **Keystream-Wiederverwendung vermeiden!** Verwende NIE denselben (Key, nonce, counter) wieder für andere Dateien. AEAD/Nonce-Disziplin ist Pflicht.
2. **Argon2-Parameter**: Standardwerte sind konservativ (512 MiB). Beim Einsatz in ressourcenbegrenzten Umgebungen anpassen — aber nicht zu klein!
3. **MAC-Verifikation**: Wenn die MAC-Verifikation fehlschlägt, verwirf die Datei. Niemals „stillschweigend“ akzeptieren.
4. **DLL-Sicherheit**: Die Geheimhaltung und Integrität von `CC_OpenCL.dll` ist wichtig — manipulierte DLLs können Keystreams offenlegen.
5. **Für echte Geheimhaltung**: Kombiniere starke Passwörter/Keydateien mit Argon2id und Nonce+MAC. Ein 256-Bit Schlüssel ist empfohlen.
6. **Nicht quantensicher formal bewiesen**: Dieses System nutzt „quantensinspirierte“ Kerne, aber das ist keine formale Post-Quantum Garantie. Siehe Risikoanalyse vor Produktionseinsatz.

---

## Troubleshooting

* `Could not find module 'CC_OpenCL.dll'`: Nutze absoluten Pfad `--dll "G:\verschluesselung\CC_OpenCL.dll"` oder lege die DLL ins Skriptverzeichnis.
* `initialize_gpu RC=...`: Manche DLL-Builds geben 1 als Erfolg; die Skripte akzeptieren 1 für `initialize_gpu`.
* → Bei anderen Fehlern: Ausgabe hier posten (inkl. Kernel-Compile Log).

---

## Weiterentwicklung (ToDo / Vorschläge)

* Header erweitern um Argon2-Parameter (m/t/p) zur exakten Reproduzierbarkeit.
* Option für Hardware-Security-Module (HSM) als Key-Quelle.
* Optionaler deterministic Counter-Mode: per Dateiblocknummer in Header.
* Benchmarks: Keystream-Durchsatz pro GPU für verschiedene Chunk-Größen.

---

## Kontakt / Credits

Entwickler: Ralf Krümmel
Projekt: CC_OpenCL / SQSE (Sub-Quantum State Encryption)



---

## Continuous Integration (GitHub Actions) — Hinweise

Eine Beispiel-Workflow-Datei für GitHub Actions wurde als `.github/workflows/ci.yml` hinzugefügt.
Der Workflow führt Unit-Tests, Paketinstallation und einen Self-Test (1 MB) aus. Hinweis: die `CC_OpenCL.dll`
ist **nicht** im Repository enthalten und kann in CI nicht ohne weiteres geladen werden.

### Mock / Test-DLL in CI
Im Beispiel-Workflow wird ein kleiner Linux-Stub (`/usr/local/bin/CC_OpenCL.dll`) als Platzhalter erstellt,
damit Import-Pfade und Funktionspfade getestet werden können. Für echte Tests oder Windows-Runner:

- **Windows Runner:** Du kannst in `.github/workflows/ci.yml` einen Schritt hinzufügen, der eine Windows-kompatible Test-DLL
  (z. B. ein signiertes Test-Artefakt) in das Arbeitsverzeichnis kopiert. Beispiel:
  ```yaml
  - name: Download test DLL (Windows)
    if: runner.os == 'Windows'
    run: |
      powershell -Command "Invoke-WebRequest -Uri 'https://example.com/your-test-CC_OpenCL.dll' -OutFile 'CC_OpenCL.dll'"
  ```

- **Sichere Handhabung:** Lade niemals deine produktive, un-signierte `CC_OpenCL.dll` in öffentliche CI-Runner hoch.
  Verwende private artifacts oder geschützte Releases (GitHub Releases with access control) oder einen Secrets-geschützten
  Storage (z. B. Azure Blob, S3) und lade die Test-DLL während des CI-Runs nur in privaten Repos.

### Wie ersetzen
1. Ersetze die Mock-Stubs im Workflow durch einen Schritt, der dein signiertes Test-DLL aus einem Artefakt-Storage lädt.  
2. Passe Pfade in `selftest` an, falls die DLL in einem anderen Pfad liegt.  
3. Führe `sqse-aead selftest-cmd` in CI nur in einem kontrollierten Test-Environment aus (nicht in public forks).

