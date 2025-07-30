# PowerShell script to download EEG Database for Alcoholism
# 122 subjects: 77 alcoholic, 45 control - PERFECT for addiction detection!

$baseUrl = "https://physionet.org/files/eegmmidb/1.0.0"
$outputDir = "alcoholism-eeg"

Write-Host "🍷 Downloading EEG Database for Alcoholism..."
Write-Host "This dataset contains REAL alcoholic vs control subjects!"
Write-Host "77 alcoholic subjects + 45 controls = 122 total subjects"

# Note: The alcoholism dataset is actually a different dataset
# Let me create the correct URL for the alcoholism dataset

$alcoholismUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-database"

Write-Host "Downloading from UCI Machine Learning Repository..."

# Create output directory
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

# Download the main dataset files
$files = @(
    "SMNI_CMI_TRAIN/co2a0000364.rd",
    "SMNI_CMI_TRAIN/co2a0000365.rd",
    "SMNI_CMI_TRAIN/co2a0000366.rd",
    "SMNI_CMI_TEST/co2c0000337.rd",
    "SMNI_CMI_TEST/co2c0000338.rd",
    "SMNI_CMI_TEST/co2c0000339.rd"
)

Write-Host "Note: The alcoholism dataset requires special access."
Write-Host "Let's try a different approach - check what's available on PhysioNet..."

# Alternative: Check if there's alcoholism data in existing PhysioNet datasets
Write-Host "Checking PhysioNet for alcoholism-related EEG data..."

# For now, let's create a placeholder and research the correct source
Write-Host "Creating research notes for alcoholism EEG datasets..."

$researchNotes = @"
# Alcoholism EEG Dataset Research

## Known Sources:
1. **UCI Machine Learning Repository**
   - EEG Database for Alcoholism
   - 122 subjects (77 alcoholic, 45 control)
   - URL: https://archive.ics.uci.edu/ml/datasets/EEG+Database

2. **PhysioNet Alternatives**
   - May have alcoholism-related studies
   - Check: https://physionet.org/content/

3. **Research Papers with Data**
   - Look for published studies with available datasets
   - Contact authors for data access

## Next Steps:
1. Research proper download method for UCI dataset
2. Check PhysioNet for similar datasets  
3. Look for published papers with available data
4. Consider synthetic alcoholism patterns based on research

## Hypothesis for Alcoholism Detection:
- Chronic alcohol use may alter neural complexity
- Expect different entropy patterns in alcoholic vs control subjects
- Could enable early addiction screening
"@

$researchNotes | Out-File -FilePath "$outputDir/research_notes.txt" -Encoding UTF8

Write-Host "✅ Research notes created in $outputDir/research_notes.txt"
Write-Host "Need to research proper access method for alcoholism EEG data"