[CmdletBinding()]
param(
    [string]$Destination = "data/raw/Processed_K50_dG_datasets.zip"
)

$ErrorActionPreference = "Stop"

$uri = "https://zenodo.org/api/records/7992926/files/Processed_K50_dG_datasets.zip/content"
$expectedMd5 = "f7e8c553efee734cf161ee6f2b0a09cf"
$destinationPath = Join-Path (Get-Location) $Destination
$partialPath = "$destinationPath.bits.part"
$jobName = "CodeCortex MegaScale v2_230420"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null

if (Test-Path -LiteralPath $destinationPath) {
    $existingMd5 = (Get-FileHash -LiteralPath $destinationPath -Algorithm MD5).Hash.ToLowerInvariant()
    if ($existingMd5 -eq $expectedMd5) {
        Write-Output "MegaScale archive already present and checksum-verified: $destinationPath"
        exit 0
    }
    throw "Existing archive checksum does not match Zenodo record 7992926. Remove it deliberately before retrying."
}

try {
    $job = Get-BitsTransfer -Name $jobName -ErrorAction Stop
    Write-Output "Resuming BITS download job $($job.JobId)"
} catch {
    Write-Output "Starting BITS download of Zenodo record 7992926 to $partialPath"
    $job = Start-BitsTransfer -Source $uri -Destination $partialPath -Asynchronous -DisplayName $jobName
}

while ($job.JobState -in @("Connecting", "Transferring", "Suspended", "Queued")) {
    Start-Sleep -Seconds 5
    $job = Get-BitsTransfer -Id $job.JobId -ErrorAction Stop
    Write-Output "BITS status: $($job.JobState) ($($job.BytesTransferred) / $($job.BytesTotal) bytes)"
}

if ($job.JobState -eq "Transferred") {
    Complete-BitsTransfer -BitsJob $job
} elseif ($job.JobState -ne "Acknowledged") {
    throw "Download failed before checksum verification: $($job.JobState). $($job.ErrorDescription)"
}

$actualMd5 = (Get-FileHash -LiteralPath $partialPath -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    Remove-Item -LiteralPath $partialPath -Force
    throw "Downloaded archive checksum mismatch: expected $expectedMd5, got $actualMd5. The partial file was removed."
}

Move-Item -LiteralPath $partialPath -Destination $destinationPath
Write-Output "Downloaded and checksum-verified: $destinationPath"
