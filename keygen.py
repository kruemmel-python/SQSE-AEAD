# keygen.py
import numpy as np, sys
size_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 64
path = sys.argv[2] if len(sys.argv) > 2 else "key.npy"
arr = np.random.default_rng().integers(0, 256, size=size_mb*1024*1024, dtype=np.uint8)
np.save(path, arr)
print(f"{path} erstellt ({size_mb} MiB)")
