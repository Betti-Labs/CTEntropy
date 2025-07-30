# PowerShell script to download CHB-MIT Epilepsy Database
# Real epilepsy patients with seizure annotations!

$baseUrl = "https://physionet.org/files/chbmit/1.0.0"
$outputDir = "chb-mit-epilepsy"

# Create output directory
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

Write-Host "Downloading CHB-MIT Epilepsy Database..."
Write-Host "This contains REAL epilepsy patients with seizure events!"

# Download for patients chb01 to chb24
for ($i = 1; $i -le 24; $i++) {
    $patient = "chb{0:D2}" -f $i
    $patientDir = Join-Path $outputDir $patient
    
    if (!(Test-Path $patientDir)) {
        New-Item -ItemType Directory -Path $patientDir
    }
    
    Write-Host "Downloading patient $patient..."
    
    try {
        # Get the patient directory listing
        $patientUrl = "$baseUrl/$patient/"
        $response = Invoke-WebRequest -Uri $patientUrl -UseBasicParsing
        
        # Parse HTML to find .edf files
        $links = $response.Links | Where-Object { $_.href -match "\.edf$" }
        
        # Limit to first 3 files per patient to save space
        $links = $links | Select-Object -First 3
        
        foreach ($link in $links) {
            $fileName = $link.href
            $fileUrl = "$patientUrl$fileName"
            $outputPath = Join-Path $patientDir $fileName
            
            # Skip if file already exists
            if (Test-Path $outputPath) {
                Write-Host "  Skipping $fileName (already exists)"
                continue
            }
            
            Write-Host "  Downloading $fileName..."
            try {
                Invoke-WebRequest -Uri $fileUrl -OutFile $outputPath
            }
            catch {
                Write-Warning "Failed to download $fileName`: $($_.Exception.Message)"
            }
        }
        
        # Also download summary file if it exists
        try {
            $summaryUrl = "$patientUrl$patient-summary.txt"
            $summaryPath = Join-Path $patientDir "$patient-summary.txt"
            if (!(Test-Path $summaryPath)) {
                Invoke-WebRequest -Uri $summaryUrl -OutFile $summaryPath
            }
        }
        catch {
            # Summary file might not exist, that's ok
        }
        
    }
    catch {
        Write-Warning "Failed to access patient $patient`: $($_.Exception.Message)"
    }
}

Write-Host "Download complete! Epilepsy dataset saved to: $outputDir"
Write-Host "This dataset contains REAL seizure events with medical annotations!"
