$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  EPUB Ceviri Araci  --  PyInstaller Build" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($dir in @("dist", "build", "epub-translator.spec")) {
    if (Test-Path $dir) {
        Write-Host "Temizleniyor: $dir" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $dir
    }
}

Write-Host "PyInstaller kuruluyor..." -ForegroundColor Gray
uv pip install pyinstaller | Out-Null

Write-Host "Build baslatiliyor..." -ForegroundColor Green
Write-Host ""

$pyiArgs = @(
    "--onedir",
    "--windowed",
    "--name", "epub-translator",
    "--add-data", "config.toml;.",
    "--paths", "src",
    "--collect-all", "customtkinter",
    "--collect-all", "transformers",
    "--collect-all", "tokenizers",
    "--collect-all", "sentencepiece",
    "--collect-all", "sacremoses",
    "--collect-submodules", "epub_translator",
    "--hidden-import", "torch",
    "--hidden-import", "torch.cuda",
    "--hidden-import", "huggingface_hub",
    "--hidden-import", "safetensors",
    "--hidden-import", "tomli_w",
    "--hidden-import", "psutil",
    "--noconfirm",
    "main.py"
)

uv run pyinstaller @pyiArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Build BASARISIZ!" -ForegroundColor Red
    exit 1
}

Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
Remove-Item -Force "epub-translator.spec" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Build tamamlandi!" -ForegroundColor Green
Write-Host "  -> dist\epub-translator\epub-translator.exe" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dagitim notu:" -ForegroundColor Yellow
Write-Host "  Modelleri dist\epub-translator\models\ icine kopyalayin." -ForegroundColor Yellow
Write-Host ""
