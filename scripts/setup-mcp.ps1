# PowerShell script to generate mcp.json from template with environment variables
# Usage: .\scripts\setup-mcp.ps1

$templatePath = ".kilocode\mcp.json.template"
$outputPath = ".kilocode\mcp.json"

# Check if template exists
if (-not (Test-Path $templatePath)) {
    Write-Host "Template file not found: $templatePath" -ForegroundColor Red
    exit 1
}

# Load .env file if it exists
$envFile = ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile..." -ForegroundColor Green
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim() -replace '^["'']|["'']$', ''
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Get environment variables
$apiKey = [Environment]::GetEnvironmentVariable("SMITHERY_API_KEY")
$profile = [Environment]::GetEnvironmentVariable("SMITHERY_PROFILE")

if (-not $apiKey) {
    Write-Host "SMITHERY_API_KEY environment variable not found!" -ForegroundColor Red
    Write-Host "Set it with: [System.Environment]::SetEnvironmentVariable('SMITHERY_API_KEY', 'your-key', [System.EnvironmentVariableTarget]::User)" -ForegroundColor Yellow
    exit 1
}

if (-not $profile) {
    Write-Host "SMITHERY_PROFILE environment variable not found!" -ForegroundColor Red
    Write-Host "Set it with: [System.Environment]::SetEnvironmentVariable('SMITHERY_PROFILE', 'your-profile', [System.EnvironmentVariableTarget]::User)" -ForegroundColor Yellow
    exit 1
}

# Read template and substitute variables
$content = Get-Content $templatePath -Raw
$content = $content -replace '\$\{SMITHERY_API_KEY\}', $apiKey
$content = $content -replace '\$\{SMITHERY_PROFILE\}', $profile

# Write to output file
$content | Out-File -FilePath $outputPath -Encoding UTF8 -NoNewline

Write-Host "Generated $outputPath with environment variables:" -ForegroundColor Green
Write-Host "  SMITHERY_API_KEY = $($apiKey.Substring(0, 8))..." -ForegroundColor Cyan
Write-Host "  SMITHERY_PROFILE = $profile" -ForegroundColor Cyan