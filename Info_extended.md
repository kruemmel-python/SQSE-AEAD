# Info — Erweiterte Analyse: Kosten, Zeit & Visualisierung

Diese Datei ergänzt die vorherige `Info.md` um konkrete Kostenabschätzungen für Angreifer, eine Visualisierung (Zeit vs Entropie) und ein GitHub Issue‑Template für Security‑Audits.

**Datum:** automatische Generierung
<img width="1024" height="1024" alt="Generated Image November 03, 2025 - 7_49PM" src="https://github.com/user-attachments/assets/6d2a8715-702e-47da-9424-2d3eb741ff6b" />

---

## 1) Annahmen für Kostenrechnung

Wir geben drei Angreifer‑Profile mit realistischen (aber vereinfachten) Annahmen:


- **Low‑cost GPU farm (cloud, non‑memory‑hard targets):**

  - Annahme: 1 GPU liefert R = 1e9 Versuche/s (nur für sehr schneller, nicht memory‑hard KDF).
  - Preis: $3 / GPU‑hour (spot/discounted).

- **Moderate attacker (on‑prem cluster):**

  - Annahme: 1 GPU liefert R = 1e8 Versuche/s; höhere Zuverlässigkeit, $1 / GPU‑hour (amortisiert)

- **Argon2‑protected target (memory‑hard)**

  - Pro Hash dauert stark länger; Beispiel: mem=512 MiB, t=3 -> per‑hash Zeit auf Angreifer‑Hardware ≈ 0.5 s → R ≈ 2 H/s per worker.


**Hinweis:** Diese Werte sind bewusst konservativ vereinfacht. Die Realität hängt stark von KDF‑Parametern, Implementierung, GPU‑Typ und Parallelisierbarkeit ab.

---

## 2) Kostenabschätzung (Beispiele)

Wir rechnen die Kosten, um im Erwartungsfall (durchschnittlich) einen Schlüssel zu finden, also T years = (2^b) / (2*R*S). Die benötigte GPU‑Stunden sind `T_years * seconds_per_year * GPUs_required`.


| Entropie (Bits) | Profile | Erwartete Jahre | Benötigte GPU‑Hours (1 GPU) | Geschätzte Kosten (1 GPU) |

|---:|---|---:|---:|---:|

| 40 | Low-cost GPU (R=1e9/s, $3/GPUh) | 0.00 years | 0 hours | $0.46 |

| 40 | Moderate (R=1e8/s, $1/GPUh) | 0.00 years | 2 hours | $1.53 |

| 40 | Argon2-heavy (R=2/s per worker, $0.05/worker-hour) | 8710.36 years | 7.635e+07 hours | $3.818e+06 |

| 56 | Low-cost GPU (R=1e9/s, $3/GPUh) | 1.14 years | 10,008 hours | $30,024.00 |

| 56 | Moderate (R=1e8/s, $1/GPUh) | 11.42 years | 100,080 hours | $100,079.99 |

| 56 | Argon2-heavy (R=2/s per worker, $0.05/worker-hour) | 5.708e+08 years | 5.004e+12 hours | $2.502e+11 |

| 64 | Low-cost GPU (R=1e9/s, $3/GPUh) | 292.27 years | 2.562e+06 hours | $7.686e+06 |

| 64 | Moderate (R=1e8/s, $1/GPUh) | 2922.71 years | 2.562e+07 hours | $2.562e+07 |

| 64 | Argon2-heavy (R=2/s per worker, $0.05/worker-hour) | 1.461e+11 years | 1.281e+15 hours | $6.405e+13 |

| 80 | Low-cost GPU (R=1e9/s, $3/GPUh) | 1.915e+07 years | 1.679e+11 hours | $5.037e+11 |

| 80 | Moderate (R=1e8/s, $1/GPUh) | 1.915e+08 years | 1.679e+12 hours | $1.679e+12 |

| 80 | Argon2-heavy (R=2/s per worker, $0.05/worker-hour) | 9.577e+15 years | 8.395e+19 hours | $4.198e+18 |


---

## 3) Interpretation

- Für Non‑memory‑hard Ziele (R sehr hoch) sind niedrige Entropien (≤64 bit) in einem kostlichen Bereich angreifbar.

- Bei Argon2‑geschützten Passwörtern ist die Rate R dramatisch kleiner (hier: 2 H/s), dadurch steigen Kosten in praktikable Bereiche für Angreifer enorm.

- Dein 65 MiB Key bleibt auch in diesen Rechnungen praktisch unknackbar.

---

## 4) Visualisierung

Die Grafik `time_vs_entropy.png` zeigt log10(Erwartete Jahre) gegen die Schlüssellänge (Bits) für verschiedene Angriffs‑Raten R. Bei Entropien über ~128 Bit ist die erwartete Zeit astronomisch.

---

## 5) Einbindung in Repo

- Datei `docs/Info.md` enthält diese Analyse (automatisch generiert).
- Ein Issue‑Template `ISSUE_TEMPLATE/security_audit.md` wurde hinzugefügt, siehe unten.

---
