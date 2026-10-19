$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = [Microsoft.Office.Interop.Word.WdAlertLevel]::wdAlertsNone

$docxPath = (Resolve-Path "output/pdf/2606MGAIL130.docx").Path
$pdfPath = [System.IO.Path]::ChangeExtension($docxPath, ".pdf")

Write-Host "Converting $docxPath to $pdfPath..."
$doc = $word.Documents.Open($docxPath)
$doc.SaveAs([ref]$pdfPath, [ref]17)
$doc.Close()
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

Write-Host "PDF generated successfully at $pdfPath"
