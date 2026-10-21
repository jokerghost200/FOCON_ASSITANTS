param (
    [string]$SourceFolder = "dist\FoconAssistant",
    [string]$ZipFileName = "FoconAssistant_Setup.zip"
)

if (Test-Path $ZipFileName) {
    Remove-Item $ZipFileName -Force
}

if (Test-Path $SourceFolder) {
    Write-Host "Création du fichier ZIP $ZipFileName..."
    Compress-Archive -Path "$SourceFolder\*" -DestinationPath $ZipFileName
    Write-Host "Terminé ! Vous pouvez distribuer le fichier $ZipFileName"
} else {
    Write-Host "Le dossier $SourceFolder n'existe pas. Veuillez exécuter PyInstaller d'abord."
}
