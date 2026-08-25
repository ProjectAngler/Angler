[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..\..'))
$failures = [System.Collections.Generic.List[string]]::new()

function Assert-File {
    param([Parameter(Mandatory)][string]$RelativePath)
    $path = Join-Path $projectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures.Add("Missing file: $RelativePath")
    }
}

function Assert-Contains {
    param(
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string[]]$Patterns
    )
    $path = Join-Path $projectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures.Add("Missing file for content check: $RelativePath")
        return
    }
    $content = Get-Content -Raw -LiteralPath $path
    foreach ($pattern in $Patterns) {
        if ($content -notmatch $pattern) {
            $failures.Add("Missing required pattern '$pattern' in $RelativePath")
        }
    }
}

$requiredFiles = @(
    'docs\blueprints\branches\safety\children\threat-model\BLUEPRINT.md',
    'docs\blueprints\branches\safety\children\human-authority\BLUEPRINT.md',
    'docs\blueprints\branches\safety\children\permissions\BLUEPRINT.md',
    'docs\blueprints\branches\safety\policies\ANG-POL-LOCAL-SCAFFOLD-001.md',
    'docs\blueprints\branches\safety\assessments\ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md',
    'docs\blueprints\branches\safety\gates\ANG-GATE-CONSTRUCTION-RELEASE-0-001.md',
    'docs\blueprints\branches\safety\contracts\ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001.md',
    'docs\blueprints\branches\safety\gates\ANG-GATE-HUMAN-FLOURISHING-001.md',
    'docs\blueprints\decisions\ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md'
)
$requiredFiles | ForEach-Object { Assert-File -RelativePath $_ }

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\policies\ANG-POL-LOCAL-SCAFFOLD-001.md' -Patterns @(
    'authorization_kind:\s*BOOTSTRAP_WORK',
    'model acquisition, loading, inference, training',
    'GPU use',
    'any network',
    'package, plugin, dependency, tool',
    'outputs/\*\*',
    'real-person data',
    'out-of-scope files',
    'background or persistent processes',
    'external side effect',
    'self-modification',
    '600 seconds per command',
    '4 GiB RAM',
    'genuinely unresolved value conflict expects `ESCALATE`',
    '5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3'
)

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\contracts\ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001.md' -Patterns @(
    'authorization_kind: PROMOTION \| BOOTSTRAP_WORK',
    '## `BOOTSTRAP_WORK` semantics',
    'not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`',
    'Any unresolved value conflict'
)

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\gates\ANG-GATE-HUMAN-FLOURISHING-001.md' -Patterns @(
    '## Bootstrap non-equivalence',
    'does \*\*not\*\* satisfy this gate',
    'not a flourishing-gate decision, Slice-00 decision, M0 decision'
)

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\assessments\ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md' -Patterns @(
    'authorization_kind:\s*BOOTSTRAP_WORK',
    'impact_class:\s*LOW',
    'disposition:\s*ALLOW',
    'human_authority:\s*ANG-AUTH-PROJECT-OWNER-001',
    'explicit instruction to complete the prerequisites for building',
    'not represented as review or approval of undisclosed material/high-impact details',
    'ANG-ADR-0002',
    'ANG-POL-LOCAL-SCAFFOLD-001',
    '5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3'
)

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\children\human-authority\BLUEPRINT.md' -Patterns @(
    'ANG-AUTH-PROJECT-OWNER-001',
    'ANG-AUTH-LEARNER-001',
    'Approve own artifact/assessment',
    'Immediate stop',
    'Resume after boundary, policy, secrecy, or authority violation'
)

Assert-Contains -RelativePath 'docs\blueprints\branches\safety\children\permissions\BLUEPRINT.md' -Patterns @(
    'Network transfer \| `0`',
    'GPU allocation/use \| `0`',
    'New packages/models/tools \| `0`',
    'Always excluded:',
    'outputs/\*\*',
    'unresolved normative conflicts expect `ESCALATE`'
)

$archivePath = Join-Path $projectRoot 'work\pre-construction-release-0-20260825.zip'
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    $failures.Add('Rollback archive is missing')
}
else {
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash
    $expectedHash = '5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3'
    if ($actualHash -ne $expectedHash) {
        $failures.Add("Rollback archive hash mismatch: $actualHash")
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output 'PASS - CR0 safety design, bootstrap non-equivalence, authority, permission ceilings, assessment, and rollback hash validated.'

