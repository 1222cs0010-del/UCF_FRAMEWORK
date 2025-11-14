<#
PowerShell helper to create the `ucf` conda environment and install PyTorch for a chosen CUDA version.
Usage (PowerShell):
  .\create_conda_env.ps1 -CudaVersion 11.8

This script assumes Miniconda/Anaconda is installed and `conda` is available in PATH.
#>
param(
    [string]$EnvFile = "..\environment.yml",
    [string]$Name = "ucf",
    [string]$CudaVersion = "11.8"
)

Write-Output "Creating conda env from $EnvFile as $Name"
conda env create -f $EnvFile -n $Name

Write-Output "Activating $Name"
conda activate $Name

Write-Output "Installing PyTorch for CUDA $CudaVersion (adjust if necessary)"
Write-Output "If this command fails, pick the correct CUDA index from https://pytorch.org/get-started/locally/"
conda install -y pytorch pytorch-cuda=$CudaVersion -c pytorch -c nvidia

Write-Output "Installing common pip packages (transformers, accelerate, datasets, peft)"
pip install --no-cache-dir transformers accelerate datasets peft sentencepiece evaluate

Write-Output "Done. If bitsandbytes is required on Windows, follow bitsandbytes documentation for Windows-specific installation." 
