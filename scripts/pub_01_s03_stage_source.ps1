param(
    [string]$Config = 'configs/pub_01_s03_strong_baselines_20261013.json',
    [string]$Destination = 'reports/pub_01/s03/strong_baselines/source_inspection_only'
)
$ErrorActionPreference = 'Stop'
$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($cfg.payload_download -or $cfg.weight_download) { throw 'Source-only staging required' }
foreach ($repo in $cfg.repos) {
    $dest = Join-Path $Destination $repo.name
    if (Test-Path -LiteralPath $dest) { throw "Existing source staging: $dest; use a fresh destination" }
    New-Item -ItemType Directory -Path $dest | Out-Null
    $treeUrl = 'https://api.github.com/repos/' + $repo.repo + '/git/trees/' + $repo.commit + '?recursive=1'
    $tree = Invoke-RestMethod -Uri $treeUrl
    if ($tree.truncated) { throw 'Truncated source tree' }
    $tree | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $dest 'tree_metadata.json') -Encoding utf8
    foreach ($file in $repo.files) {
        if ($file -match '(^|/)\.\.(/|$)' -or $file -match '\.(pkl|npz|npy|pt|ckpt|wav|bdf|bin|safetensors)$') {
            throw 'Payload or unsafe path forbidden'
        }
        $path = Join-Path $dest $file
        New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
        $url = 'https://raw.githubusercontent.com/' + $repo.repo + '/' + $repo.commit + '/' + $file
        Invoke-WebRequest -Uri $url -OutFile $path
    }
    Write-Output ('Staged source text only: ' + $repo.name)
}
