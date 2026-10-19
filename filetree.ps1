param(
    [string]$OutputFile = (Join-Path $PSScriptRoot "filetree_out.txt")
)

# Set console output encoding to UTF-8 so tree characters render cleanly in terminal
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Folders to print but not recurse into
$skip_folders = @(".venv", "__pycache__", "Logs", "_backups", "temp", "archive", "favicon_io", "content")

function Get-Tree {
    param(
        [string]$Path,
        [string]$Prefix = ""
    )

    $items = Get-ChildItem -Path $Path | Sort-Object @{Expression = { -not $_.PSIsContainer }}, Name

    for ($i = 0; $i -lt $items.Count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq ($items.Count - 1))

        if ($isLast) {
            $connector = [char]0x2514 + [char]0x2500 + [char]0x2500 + " "
            $newPrefix = $Prefix + "    "
        } else {
            $connector = [char]0x251C + [char]0x2500 + [char]0x2500 + " "
            $newPrefix = $Prefix + [char]0x2502 + "   "
        }

        if ($item.PSIsContainer) {
            "$Prefix$connector$($item.Name)"
        } else {
            $size = if ($item.Length -ge 1GB) {
                "{0:N2} GB" -f ($item.Length / 1GB)
            } elseif ($item.Length -ge 1MB) {
                "{0:N2} MB" -f ($item.Length / 1MB)
            } elseif ($item.Length -ge 1KB) {
                "{0:N2} KB" -f ($item.Length / 1KB)
            } else {
                "$($item.Length) B"
            }
            $lastModified = $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            "$Prefix$connector$($item.Name) | $size | $lastModified"
        }

        if ($item.PSIsContainer -and ($item.Name -notin $script:skip_folders)) {
            Get-Tree -Path $item.FullName -Prefix $newPrefix
        }
    }
}

$root = Get-Item .
$lines = @()
$lines += $root.Name
$lines += Get-Tree -Path $root.FullName -Prefix ""

$lines | Set-Content -Path $OutputFile -Encoding UTF8
$lines
