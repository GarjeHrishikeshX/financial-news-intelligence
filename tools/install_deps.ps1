# Install Python dependencies for the project
# Usage: Open PowerShell (run as Administrator if needed) and execute:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#   .\tools\install_deps.ps1

# Set PYTHONPATH for the current process
$env:PYTHONPATH = Get-Location

Write-Host "Installing packages from requirements.txt..."
python -m pip install -r requirements.txt

Write-Host "Installing CPU-only PyTorch and sentence-transformers (recommended for CPU machines)..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers

Write-Host "Done. You can now run tests or start the server."
Write-Host "If you have a CUDA-enabled GPU, visit https://pytorch.org for CUDA-specific instructions."
