# PowerShell script to load environment variables from .env file
# Usage: .\scripts\load-env.ps1

$envFile = ".env"

if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile..." -ForegroundColor Green
    
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            
            # Set environment variable for current session
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "  $name = $value" -ForegroundColor Cyan
        }
    }
    
    Write-Host "Environment variables loaded successfully!" -ForegroundColor Green
} else {
    Write-Host "No .env file found. Create one from .env.template" -ForegroundColor Yellow
    Write-Host "  cp .env.template .env" -ForegroundColor Gray
}