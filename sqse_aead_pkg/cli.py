
from __future__ import annotations
import typer
from pathlib import Path
from rich import print as rprint
from .core import encrypt_file, decrypt_file, selftest, DEFAULT_CHUNK, SQSEAEADError

app = typer.Typer(help="SQSE AEAD – GPU/DLL-gestützte Verschlüsselung (Argon2/Seed/NPY, Nonce/Counter, BLAKE3-MAC).")

@app.command()
def encrypt(
    infile: Path = typer.Argument(..., exists=True, readable=True),
    outfile: Path = typer.Argument(...),
    dll: str = typer.Option("CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll"),
    gpu: int = typer.Option(0, help="GPU-Index"),
    K: float = typer.Option(1.2, help="chaos_K"),
    steps: int = typer.Option(16, help="Iterationen"),
    chunk_bytes: int = typer.Option(DEFAULT_CHUNK, help="Chunkgröße in Bytes"),
    password: str = typer.Option(None, "--pass", help="Passwort für Argon2id"),
    seed: int = typer.Option(None, help="Deterministischer Seed"),
    key_npy: Path = typer.Option(None, help="Pfad zu .npy für Rohkey"),
    argon_mem_mb: int = typer.Option(512, help="Argon2 memory (MiB)"),
    argon_t: int = typer.Option(3, help="Argon2 time_cost"),
    argon_p: int = typer.Option(4, help="Argon2 parallelism"),
):
    try:
        encrypt_file(infile, outfile, dll, gpu, K, steps, password, seed, key_npy, chunk_bytes, argon_mem_mb, argon_t, argon_p)
        rprint(f"[bold green]OK[/]: Encrypted: {infile} -> {outfile}")
    except SQSEAEADError as e:
        rprint(f"[bold red]Fehler[/]: {e}")
        raise typer.Exit(code=1)

@app.command()
def decrypt(
    infile: Path = typer.Argument(..., exists=True, readable=True),
    outfile: Path = typer.Argument(...),
    dll: str = typer.Option("CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll"),
    gpu: int = typer.Option(0, help="GPU-Index"),
    password: str = typer.Option(None, "--pass", help="Passwort für Argon2id"),
    seed: int = typer.Option(None, help="Deterministischer Seed"),
    key_npy: Path = typer.Option(None, help="Pfad zu .npy für Rohkey"),
):
    try:
        decrypt_file(infile, outfile, dll, gpu, password, seed, key_npy)
        rprint(f"[bold green]OK[/]: Decrypted: {infile} -> {outfile}")
    except SQSEAEADError as e:
        rprint(f"[bold red]Fehler[/]: {e}")
        raise typer.Exit(code=1)

@app.command()
def selftest_cmd(
    dll: str = typer.Option("CC_OpenCL.dll", help="Pfad zur CC_OpenCL.dll"),
    gpu: int = typer.Option(0, help="GPU-Index"),
    K: float = typer.Option(1.2, help="chaos_K"),
    steps: int = typer.Option(16, help="Iterationen"),
    seed: int = typer.Option(1234, help="Seed für reproduzierbaren Test"),
    size: int = typer.Option(1_000_000, help="Testgröße in Bytes (z. B. 1_000_000 = ~1 MB)"),
):
    ok, msg = selftest(dll, gpu, K, steps, seed, size)
    if ok:
        rprint(f"[bold green]{msg}[/]")
    else:
        rprint(f"[bold red]{msg}[/]")
        raise typer.Exit(code=2)
