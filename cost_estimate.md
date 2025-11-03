# Cost Estimation: How expensive is an attack? (rough example)

This document provides a simple, order-of-magnitude estimation of what an attacker would need
to invest to mount a brute-force attack on a 256-bit key or on a weaker password under Argon2.

## Assumptions
- Hardware & electricity costs vary; we give rough numbers.
- For Argon2 with heavy memory (512 MiB) the attacker cannot use simple GPU farms cheaply.

### Example: attacking a 32-bit password-space (bad)
- 2^32 ≈ 4.29e9 guesses. At $0.01 per 1e6 guesses (cheap cloud GPU), cost ≈ $43.

### Example: attacking a 56-bit key-space
- 2^56 ≈ 7.2e16 guesses. At 1e9 guesses/s, time ≈ 2.28e8 s ≈ 7.2 years; cost for such infra >> $ millions.

### Example: attacking Argon2-protected password
- If each Argon2 invocation with mem=512MB, t=3 takes 0.5s on attacker hardware, then guesses/s ≈ 2.
  To reach 1e9 guesses you'd need enormous parallelism (≈5e8 parallel workers), utterly unreal.

## Takeaway
- Protect the key material; avoid relying on brute-force resistance alone for human-chosen passwords.
- Use large random `.npy` keys or strong passphrases + Argon2.
