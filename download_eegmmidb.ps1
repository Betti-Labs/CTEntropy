# PowerShell script to download EEG Motor Movement/Imagery Database
# Equivalent to: wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

$baseUrl = "https://physionet.org/files/eegmmidb/1.0.0"
$outputDir = "eegmmidb"

# Create output directory
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

Write-Host "Starting download of EEG Motor Movement/Imagery Database..."
Write-Host "This will download data for 109 subjects - it may take a while!"

# Download for subjects S001 to S109
for ($i = 1; $i -le 109; $i++) {
    $subject = "S{0:D3}" -f $i
    $subjectDir = Join-Path $outputDir $subject
    
    if (!(Test-Path $subjectDir)) {
        New-Item -ItemType Directory -Path $subjectDir
    }
    
    Write-Host "Downloading subject $subject..."
    
    try {
        # Get the subject directory listing
        $subjectUrl = "$baseUrl/$subject/"
        $response = Invoke-WebRequest -Uri $subjectUrl -UseBasicParsing
        
        # Parse HTML to find .edf files
        $links = $response.Links | Where-Object { $_.href -match "\.edf$" }
        
        foreach ($link in $links) {
            $fileName = $link.href
            $fileUrl = "$subjectUrl$fileName"
            $outputPath = Join-Path $subjectDir $fileName
            
            # Skip if file already exists (resume capability)
            if (Test-Path $outputPath) {
                Write-Host "  Skipping $fileName (already exists)"
                continue
            }
            
            Write-Host "  Downloading $fileName..."
            try {
                Invoke-WebRequest -Uri $fileUrl -OutFile $outputPath
            }
            catch {
                Write-Warning "Failed to download $fileName`: $_"
            }
        }
    }
    catch {
        Write-Warning "Failed to access subject $subject`: $_"
    }
}

Write-Host "Download complete! Dataset saved to: $outputDir"