\
<#
apply_patch.ps1
Usage: Run from the root of your git repository in PowerShell (Windows).
This script copies the provided files into the current repo, stages, commits and optionally pushes.
It will not force-push or change remote settings.
#>

param(
    [switch]$Push = $false,
    [string]$AuthorName = "Ralf Krümmel",
    [string]$AuthorEmail = "you@example.com",
    [string]$CommitMessage = "Add CI workflow and security helpers (validate_header, rotate script, HSM notes, cost estimate)"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Working directory (repo root):" (Get-Location).Path
Write-Host "Copying files from packaged 'repo patch' to current repo..." -ForegroundColor Cyan

# Files to copy (relative paths inside zip)
$files = @(
    ".github\workflows\ci.yml",
    "validate_header.py",
    "rotate_keys_and_reencrypt.py",
    "HSM_README.md",
    "cost_estimate.md",
    "README.md"
)

foreach ($file in $files) {
    $src = Join-Path $scriptDir $file
    $dst = Join-Path (Get-Location).Path $file
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path $src)) {
        Write-Host "WARNING: Source file not found in patch package: $src" -ForegroundColor Yellow
        continue
    }
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    Copy-Item -Path $src -Destination $dst -Force
    Write-Host "Copied $file" -ForegroundColor Green
}

# Git operations
# check that we are inside a git repo
if (-not (Test-Path .git)) {
    Write-Host "ERROR: Current directory does not look like a git repository (no .git folder)." -ForegroundColor Red
    exit 2
}

# Stage files
git add .github/workflows/ci.yml validate_header.py rotate_keys_and_reencrypt.py HSM_README.md cost_estimate.md README.md
if ($LASTEXITCODE -ne 0) {
    Write-Host "git add failed." -ForegroundColor Red
    exit 3
}

# Commit
git commit -m $CommitMessage --author="$AuthorName <$AuthorEmail>"
if ($LASTEXITCODE -ne 0) {
    Write-Host "git commit failed or nothing to commit." -ForegroundColor Yellow
} else {
    Write-Host "Committed changes." -ForegroundColor Green
    if ($Push) {
        Write-Host "Pushing to remote..." -ForegroundColor Cyan
        git push
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git push failed." -ForegroundColor Red
            exit 4
        }
    } else {
        Write-Host "Not pushing. Use -Push to push to remote." -ForegroundColor Cyan
    }
}

Write-Host "Done." -ForegroundColor Green
