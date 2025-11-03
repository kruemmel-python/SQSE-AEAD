#!/usr/bin/env python3
"""rotate_keys_and_reencrypt.py

Re-encrypt all files in a directory from an old key to a new key.
Usage (example):
  python rotate_keys_and_reencrypt.py --dir encrypted_store --old-seed 1234 --new-pass "NewSecret!" --dll CC_OpenCL.dll
Notes:
  - This script is cautious: decrypts to temporary file, re-encrypts to temporary, verifies bit-equality, then atomically renames.
  - Parallelism via workers (multiprocessing). Use with caution and test first.
"""
from __future__ import annotations
import argparse, tempfile, shutil, os, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from sqse_aead_pkg.core import decrypt_file, encrypt_file, SQSEAEADError

def process_file(path: Path, dll: str, gpu: int, old_args: dict, new_args: dict, chunk_bytes: int):
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_plain = Path(td) / (path.stem + '.plain')
            tmp_new = Path(td) / (path.stem + '.new.aead')
            # decrypt using old key
            decrypt_file(path, tmp_plain, dll, gpu, old_args.get('password'), old_args.get('seed'), old_args.get('key_npy'))
            # encrypt with new key
            encrypt_file(tmp_plain, tmp_new, dll, gpu, new_args.get('K',1.2), new_args.get('steps',16),
                         new_args.get('password'), new_args.get('seed'), new_args.get('key_npy'),
                         chunk_bytes, new_args.get('argon_mem_mb',512), new_args.get('argon_t',3), new_args.get('argon_p',4))
            # verify by decrypting new to verify file
            tmp_verify = Path(td) / (path.stem + '.verify')
            decrypt_file(tmp_new, tmp_verify, dll, gpu, new_args.get('password'), new_args.get('seed'), new_args.get('key_npy'))
            if tmp_verify.read_bytes() != tmp_plain.read_bytes():
                return (path, False, 'verification_failed')
            # atomic replace: move new to original location with .rotated suffix then rename
            backup = path.with_suffix(path.suffix + '.bak')
            path.rename(backup)
            tmp_new.rename(path)
            backup.unlink()
            return (path, True, 'rotated')
    except Exception as e:
        return (path, False, str(e))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Directory with .aead files')
    p.add_argument('--dll', default='CC_OpenCL.dll')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--chunk-bytes', type=int, default=8*1024*1024)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--old-pass', dest='old_pass')
    group.add_argument('--old-seed', dest='old_seed', type=int)
    group.add_argument('--old-key-npy', dest='old_key_npy')
    ngroup = p.add_mutually_exclusive_group(required=True)
    ngroup.add_argument('--new-pass', dest='new_pass')
    ngroup.add_argument('--new-seed', dest='new_seed', type=int)
    ngroup.add_argument('--new-key-npy', dest='new_key_npy')

    p.add_argument('--workers', type=int, default=max(1, cpu_count()-1))
    args = p.parse_args()

    old_args = {'password': args.old_pass, 'seed': args.old_seed, 'key_npy': Path(args.old_key_npy) if args.old_key_npy else None}
    new_args = {'password': args.new_pass, 'seed': args.new_seed, 'key_npy': Path(args.new_key_npy) if args.new_key_npy else None, 'K':1.2, 'steps':16, 'argon_mem_mb':512, 'argon_t':3, 'argon_p':4}
    files = list(Path(args.dir).glob('**/*.aead'))
    if not files:
        print('No .aead files found in', args.dir); sys.exit(2)
    print(f'Rotating {len(files)} files using {args.workers} workers...')

    fn = partial(process_file, dll=args.dll, gpu=args.gpu, old_args=old_args, new_args=new_args, chunk_bytes=args.chunk_bytes)
    with Pool(args.workers) as pool:
        for path, ok, msg in pool.imap_unordered(fn, files):
            print(path, ok, msg)

if __name__ == '__main__':
    main()
