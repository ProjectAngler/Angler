[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Errors = [System.Collections.Generic.List[string]]::new()

function Add-ValidationError {
    param([Parameter(Mandatory)][string]$Message)
    $script:Errors.Add($Message)
}

function Get-Sha256 {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
}

function Get-Utf8TextSha256 {
    param([Parameter(Mandatory)][string]$Text)
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
    return [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($bytes))
}

function Assert-FileHash {
    param(
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$ExpectedHash
    )
    $fullPath = Join-Path $script:RepoRoot $RelativePath
    $actual = Get-Sha256 -Path $fullPath
    if ($null -eq $actual) {
        Add-ValidationError "Missing immutable file: $RelativePath"
    }
    elseif ($actual -ne $ExpectedHash.ToUpperInvariant()) {
        Add-ValidationError "Hash mismatch for ${RelativePath}: expected $ExpectedHash, found $actual"
    }
}

function Get-FrontMatterBlock {
    param([Parameter(Mandatory)][string]$Raw)
    $match = [regex]::Match($Raw, '\A---\r?\n(?<body>.*?)\r?\n---(?:\r?\n|\z)', [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $match.Success) {
        return $null
    }
    return $match.Groups['body'].Value
}

function Get-ScalarField {
    param(
        [Parameter(Mandatory)][string]$FrontMatter,
        [Parameter(Mandatory)][string]$Name
    )
    $match = [regex]::Match($FrontMatter, "(?m)^$([regex]::Escape($Name)):\s*(?<value>.*?)\s*$")
    if (-not $match.Success) {
        return $null
    }
    return $match.Groups['value'].Value.Trim().Trim('"').Trim("'")
}

function Get-ListField {
    param(
        [Parameter(Mandatory)][string]$FrontMatter,
        [Parameter(Mandatory)][string]$Name
    )
    $lines = @($FrontMatter -split '\r?\n')
    $values = [System.Collections.Generic.List[string]]::new()
    $collecting = $false
    foreach ($line in $lines) {
        if ($line -match "^$([regex]::Escape($Name)):\s*$") {
            $collecting = $true
            continue
        }
        if ($collecting) {
            if ($line -match '^\s{2}-\s+(?<value>.+?)\s*$') {
                $values.Add($Matches['value'].Trim().Trim('"').Trim("'"))
                continue
            }
            if ($line -match '^\S') {
                break
            }
        }
    }
    return @($values)
}

function Assert-ScalarFields {
    param(
        [Parameter(Mandatory)][string]$FrontMatter,
        [Parameter(Mandatory)][hashtable]$Expected,
        [Parameter(Mandatory)][string]$Label
    )
    foreach ($name in $Expected.Keys) {
        $actual = Get-ScalarField -FrontMatter $FrontMatter -Name $name
        if ($actual -ne [string]$Expected[$name]) {
            Add-ValidationError "$Label field '$name' expected '$($Expected[$name])', found '$actual'"
        }
    }
}

function Assert-ExactSet {
    param(
        [Parameter(Mandatory)][object[]]$Actual,
        [Parameter(Mandatory)][object[]]$Expected,
        [Parameter(Mandatory)][string]$Label
    )
    $actualStrings = @($Actual | ForEach-Object { [string]$_ })
    $expectedStrings = @($Expected | ForEach-Object { [string]$_ })
    if ($actualStrings.Count -ne @($actualStrings | Sort-Object -Unique).Count) {
        Add-ValidationError "$Label contains duplicate entries"
    }
    $missing = @($expectedStrings | Where-Object { $_ -notin $actualStrings })
    $extra = @($actualStrings | Where-Object { $_ -notin $expectedStrings })
    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        Add-ValidationError "$Label exact-set mismatch; missing=[$($missing -join ', ')]; extra=[$($extra -join ', ')]"
    }
}

function Require-LiteralText {
    param(
        [Parameter(Mandatory)][string]$Raw,
        [Parameter(Mandatory)][string[]]$Values,
        [Parameter(Mandatory)][string]$Label
    )
    foreach ($value in $Values) {
        if (-not $Raw.Contains($value)) {
            Add-ValidationError "$Label is missing required literal: $value"
        }
    }
}

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$releaseRoot = Join-Path $script:RepoRoot 'docs\blueprints\releases\construction-0'
$manifestPath = Join-Path $releaseRoot 'MANIFEST.md'
$baselinePath = Join-Path $releaseRoot 'baselines\ANG-BASELINE-CR0-CONTINUITY-001.json'
$gatePath = Join-Path $releaseRoot 'gates\ANG-GATE-CR0-CONTINUITY-001.md'
$leafPath = Join-Path $script:RepoRoot 'docs\blueprints\work\slice-00\ANG-WORK-CR0-CONTINUITY-001.md'
$specPath = Join-Path $releaseRoot 'revalidations\ANG-CR0-REVALIDATION-20260825-005.md'
$decisionPath = Join-Path $releaseRoot 'revalidations\ANG-CR0-REVALIDATION-20260825-005-decision.json'
$receiptPath = Join-Path $releaseRoot 'branch-receipts\CONTINUITY.md'
$selfPath = $MyInvocation.MyCommand.Path

$expectedBaseCommit = '7f383939d021c4bba9dd5af046ce0838b032ff02'
$expectedBaselineHash = 'EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7'
$expectedGateHash = '85DAEB5E3FE72FE37EF40A141FB7DA8B2A133590C6520B2BC1BC55D0C94E20AC'
$expectedLeafHash = '8175422997DAF86D287A776E45BED2D7F48F3BA79D3EBC65D42130A222419116'

$immutableHashes = [ordered]@{
    'docs/blueprints/decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md' = 'C5B97294FD53AFA9F95E0C28AD6F36C9A7861DF07B50B714F489ED1F37873753'
    'docs/blueprints/branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md' = '23D04D544208C7273BA6C7860CC788CDD81640C8DD8236FFD1FED1F2D77495C6'
    'docs/blueprints/branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md' = '181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC'
    'docs/blueprints/branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md' = 'B768F7669241A0C3432E95E0DDB900AE5A007B2369AB3E4D073F473434DE8EEB'
    'docs/blueprints/branches/evidence/children/evidence-schemas/work/ANG-WORK-EVIDENCE-SCHEMAS-001.md' = '5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289'
    'docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md' = 'A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5'
    'docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-EVIDENCE-SCHEMAS-001.md' = 'CCDB0782B520328AA5B0A04C6684E16EB9390B8338B61FC6BBA1CB8913A49210'
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-EVIDENCE-SCHEMAS-001.json' = 'F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F'
    'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json' = '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0'
    'artifacts/control-plane/evidence-schemas/test-receipt.json' = '897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53'
    'artifacts/control-plane/evidence-schemas/effect-receipt.json' = '9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893'
    'artifacts/control-plane/evidence-schemas/HANDOFF.md' = '017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003.md' = '1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003-decision.json' = '9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A'
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-001.json' = 'EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004.md' = '50ED004747A4DABD9D306E8931D8BFD8A558F0CDE441596645A8E028F15CF9D2'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004-decision.json' = '12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9'
    'docs/blueprints/work/slice-00/ANG-WORK-CR0-RESOURCES-001.md' = '67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA'
    'docs/blueprints/branches/resources/gates/ANG-GATE-CR0-RESOURCES-001.md' = 'A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2'
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-002.json' = 'A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE'
    'docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md' = 'D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B'
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md' = '916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A'
    'docs/blueprints/work/slice-00/validate-construction-release-0.ps1' = '50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94'
    'tools/validate_blueprint_tree.ps1' = '421A06CCE6B5528F67ADF5869F1BB578C1F4B252785CEA96434BA7C6C4CD2BE7'
}

foreach ($entry in $immutableHashes.GetEnumerator()) {
    Assert-FileHash -RelativePath $entry.Key -ExpectedHash $entry.Value
}

Assert-FileHash -RelativePath 'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-001.json' -ExpectedHash $expectedBaselineHash
Assert-FileHash -RelativePath 'docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-001.md' -ExpectedHash $expectedGateHash
Assert-FileHash -RelativePath 'docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-001.md' -ExpectedHash $expectedLeafHash

$expectedProjectionTargets = [ordered]@{
    'docs/blueprints/ROOT_CAPSULE.md' = @{ state = 'present'; sha256 = 'E2DE70E2C118432A0B8B35D7F3B26E6DA116E99C5C5AC12B9192868366D00B72'; class = 'restore_exact' }
    'docs/blueprints/STATUS.md' = @{ state = 'present'; sha256 = 'CBA795AFF3CA2C24E1C9373192E792F5B108D7390DF59294D3C6FEF675A11A54'; class = 'restore_exact' }
    'docs/blueprints/branches/resources/CAPSULE.md' = @{ state = 'present'; sha256 = '03DC17C85CF1076D8DB7482136A403D9EE4F7C804361C68210CC3232B2030175'; class = 'restore_exact' }
    'docs/blueprints/branches/resources/STATUS.md' = @{ state = 'present'; sha256 = 'A1A221EF1FBB85EF8A41D6ECC627E8FECA2290073F849834891EDFCF92465DEA'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/BLUEPRINT.md' = @{ state = 'present'; sha256 = '56DCE997BF6F002BB9202C144913B90D2A28A64203885B8A3730671AFB16ED48'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/CAPSULE.md' = @{ state = 'present'; sha256 = '54860217CACB6DC5DEADA3763F8E35371C879BD8A0A792579211B3011EE70DD3'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/STATUS.md' = @{ state = 'present'; sha256 = '118B3D928F93F3C53B299D352B6CC5A4F2D323A2CF1297B9A06C9B0CABE34DA5'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md' = @{ state = 'present'; sha256 = '2E50B0BB3F016AC0FEA3B5E72FFCDE7E945BD1D82DDA553722986339FF1F93FB'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md' = @{ state = 'present'; sha256 = 'A635F094C7C8F6A3FEEEF4D38918881CED1C50A12B07892D430A0D351424B90F'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md' = @{ state = 'present'; sha256 = '48CAC7F77ED0B1FF6A51E73F22C2E2E4162520204BF82A69DD9B40E34B4746FA'; class = 'restore_exact' }
    'docs/blueprints/BLUEPRINT_INDEX.json' = @{ state = 'present'; sha256 = '1A4EA38741243E24EF46B74DA78EDDE256BDE7E9BE7EDDDAAFA013CF7FF17015'; class = 'restore_exact' }
    'docs/blueprints/TREE.md' = @{ state = 'present'; sha256 = '851F849BC738AACAC17319C2210364218CEC3B499F13A33144F820A09098F195'; class = 'restore_exact' }
    'docs/blueprints/TRACEABILITY.md' = @{ state = 'present'; sha256 = 'B19C6F66B60E4C0EE49A4FE61E3F592C9C90DCF434EE01C662C609A828C9F6F0'; class = 'restore_exact' }
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1' = @{ state = 'absent'; sha256 = $null; class = 'restore_or_remove' }
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-001.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
}

try {
    $baseline = Get-Content -Raw -LiteralPath $baselinePath | ConvertFrom-Json
    if ($baseline.baseline_id -ne 'ANG-BASELINE-CR0-CONTINUITY-001' -or $baseline.base_commit -ne $expectedBaseCommit -or [int]$baseline.leaf_revision -ne 1) {
        Add-ValidationError 'Continuity baseline identity/base/leaf revision mismatch'
    }
    $baselineTargets = @($baseline.targets)
    if ($baselineTargets.Count -ne 17) {
        Add-ValidationError "Continuity baseline must declare 17 targets; found $($baselineTargets.Count)"
    }
    Assert-ExactSet -Actual @($baselineTargets.path) -Expected @($expectedProjectionTargets.Keys) -Label 'Continuity baseline targets'
    foreach ($target in $baselineTargets) {
        $relative = [string]$target.path
        if (-not $expectedProjectionTargets.Contains($relative)) {
            continue
        }
        $expected = $expectedProjectionTargets[$relative]
        if ([string]$target.state -ne [string]$expected.state) {
            Add-ValidationError "Baseline state mismatch for $relative"
        }
        if ($expected.state -eq 'present' -and ([string]$target.sha256).ToUpperInvariant() -ne [string]$expected.sha256) {
            Add-ValidationError "Baseline SHA-256 mismatch for $relative"
        }
    }
    foreach ($className in @('restore_exact', 'restore_or_remove', 'preserve_on_failure')) {
        $expectedMembers = @($expectedProjectionTargets.Keys | Where-Object { $expectedProjectionTargets[$_].class -eq $className })
        $actualMembers = @($baseline.rollback_classes.$className)
        Assert-ExactSet -Actual $actualMembers -Expected $expectedMembers -Label "Baseline rollback class $className"
    }
}
catch {
    Add-ValidationError "Unable to parse continuity baseline: $($_.Exception.Message)"
}

# This validator is pre-start only: both PENDING and authorized branches require the exact baseline state.
foreach ($relative in $expectedProjectionTargets.Keys) {
    $expected = $expectedProjectionTargets[$relative]
    $fullPath = Join-Path $script:RepoRoot $relative
    if ($expected.state -eq 'present') {
        $actualHash = Get-Sha256 -Path $fullPath
        if ($actualHash -ne [string]$expected.sha256) {
            Add-ValidationError "Pre-start projection does not match known baseline: $relative"
        }
    }
    elseif (Test-Path -LiteralPath $fullPath) {
        Add-ValidationError "Pre-start target must be absent: $relative"
    }
}

$knownStaleChecks = [ordered]@{
    'docs/blueprints/ROOT_CAPSULE.md' = @('ANG-CR0-REVALIDATION-20260825-004', 'PENDING')
    'docs/blueprints/STATUS.md' = @('ANG-CR0-REVALIDATION-20260825-004', 'PENDING/NON-AUTHORIZING')
    'docs/blueprints/branches/resources/CAPSULE.md' = @('ANG-CR0-REVALIDATION-20260825-004', 'PENDING/NON-AUTHORIZING')
    'docs/blueprints/branches/resources/STATUS.md' = @('revision_3_ready_pending_revalidation_004_non_authorizing', 'no Resource run, receipt')
    'docs/blueprints/branches/evidence/BLUEPRINT.md' = @('delivery_status: ready', 'Activate `ANG-WORK-EVIDENCE-SCHEMAS-001`')
    'docs/blueprints/branches/evidence/CAPSULE.md' = @('active PENDING successor `ANG-CR0-REVALIDATION-20260825-004`')
    'docs/blueprints/branches/evidence/STATUS.md' = @('active Resources successor `ANG-CR0-REVALIDATION-20260825-004`')
    'docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md' = @('delivery_status: ready', 'ANG-WORK-EVIDENCE-SCHEMAS-001` is the exact next leaf')
    'docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md' = @('No code, receipt, or gate decision exists')
    'docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md' = @('Delivery: ready', 'No decision exists yet')
    'docs/blueprints/BLUEPRINT_INDEX.json' = @('"id": "ANG-BP-EVIDENCE-SCHEMAS"', '"delivery_status": "ready"')
    'docs/blueprints/TREE.md' = @('first schema leaf ready', 'Only the exact EVIDENCE-SCHEMAS work leaf is initially ready')
    'docs/blueprints/TRACEABILITY.md' = @('ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001 (CR0-only disposition)')
}
foreach ($entry in $knownStaleChecks.GetEnumerator()) {
    $raw = Get-Content -Raw -LiteralPath (Join-Path $script:RepoRoot $entry.Key)
    Require-LiteralText -Raw $raw -Values @($entry.Value) -Label "Known stale pre-state $($entry.Key)"
}

# Verify the immutable Evidence receipt's complete 14-file implementation map.
try {
    $evidenceReceipt = Get-Content -Raw -LiteralPath (Join-Path $script:RepoRoot 'artifacts\control-plane\evidence-schemas\test-receipt.json') | ConvertFrom-Json
    $artifactProperties = @($evidenceReceipt.artifact_hashes.PSObject.Properties)
    if ($artifactProperties.Count -ne 14) {
        Add-ValidationError "Evidence test receipt must bind 14 implementation artifacts; found $($artifactProperties.Count)"
    }
    foreach ($property in $artifactProperties) {
        Assert-FileHash -RelativePath $property.Name -ExpectedHash ([string]$property.Value)
    }
}
catch {
    Add-ValidationError "Unable to validate Evidence artifact map: $($_.Exception.Message)"
}

try {
    $evidenceDecision = Get-Content -Raw -LiteralPath (Join-Path $script:RepoRoot 'artifacts\control-plane\evidence-schemas\scaffold-gate-decision.json') | ConvertFrom-Json
    if ($evidenceDecision.disposition -ne 'SCAFFOLD_ACCEPTED' -or $evidenceDecision.bindings.release_manifest.sha256 -ne '802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2') {
        Add-ValidationError 'Evidence decision disposition or execution-time Manifest-v2 binding changed'
    }
    if ($evidenceDecision.bindings.test_receipt.sha256 -ne '897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53' -or
        $evidenceDecision.bindings.effect_receipt.sha256 -ne '9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893' -or
        $evidenceDecision.bindings.handoff.sha256 -ne '017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855') {
        Add-ValidationError 'Evidence decision receipt/handoff bindings changed'
    }
    if ($evidenceDecision.non_equivalence.normal_evidence_schema_gate.status -ne 'NOT_RUN' -or $evidenceDecision.non_equivalence.human_flourishing_gate.status -ne 'NOT_RUN') {
        Add-ValidationError 'Evidence decision normal/Human-Flourishing non-equivalence changed'
    }
}
catch {
    Add-ValidationError "Unable to validate Evidence decision: $($_.Exception.Message)"
}

$resourceOutputHashes = [ordered]@{
    'schemas/control/v1/resources/resource-inventory.schema.json' = '92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F'
    'schemas/control/v1/resources/execution-plan.schema.json' = '7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2'
    'tests/synthetic/slice00/resources/constrained.inventory.json' = 'CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956'
    'tests/synthetic/slice00/resources/workstation.inventory.json' = '162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9'
    'tests/synthetic/slice00/resources/cluster.inventory.json' = 'AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85'
    'tests/synthetic/slice00/resources/invalid-overcommitted.plan.json' = '92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A'
    'tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1' = '047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA'
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md' = '916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A'
}
foreach ($entry in $resourceOutputHashes.GetEnumerator()) {
    Assert-FileHash -RelativePath $entry.Key -ExpectedHash $entry.Value
}
$resourcesRaw = Get-Content -Raw -LiteralPath (Join-Path $script:RepoRoot 'docs\blueprints\releases\construction-0\branch-receipts\RESOURCES.md')
Require-LiteralText -Raw $resourcesRaw -Values @(
    'disposition: SCAFFOLD_ACCEPTED',
    'SHA-256 `35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41`',
    'SHA-256 `12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9`',
    '`ANG-GATE-RESOURCE-DESIGN-001`: `NOT_RUN`',
    '`ANG-GATE-HUMAN-FLOURISHING-001`: `NOT_RUN`',
    'Slice 00: `NOT_PASSED`',
    'M0: `NOT_PASSED`'
) -Label 'Resources accepted receipt'

$gateRaw = Get-Content -Raw -LiteralPath $gatePath
$gateFront = Get-FrontMatterBlock -Raw $gateRaw
if ([regex]::IsMatch($gateRaw, '`(?:ACCEPTED|REJECTED)`')) {
    Add-ValidationError 'Continuity gate uses an unregistered generic disposition alias'
}
if ($null -eq $gateFront) {
    Add-ValidationError 'Continuity gate front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $gateFront -Label 'Continuity gate' -Expected @{
        gate_id = 'ANG-GATE-CR0-CONTINUITY-001'
        version = '1'
        status = 'specified'
        activation_state = 'unusable_pending_revalidation'
        activation_revalidation = 'ANG-CR0-REVALIDATION-20260825-005'
        release_manifest_version = '3'
        leaf = 'ANG-WORK-CR0-CONTINUITY-001@1'
        baseline_sha256 = $expectedBaselineHash
        executor = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001'
        executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        independent_verifier = 'ANG-AUTH-VALIDATOR-001'
        independent_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001'
        independent_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        allowed_dispositions = 'RECONCILIATION_ACCEPTED|RECONCILIATION_REJECTED|ESCALATE'
        decision_path = 'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md'
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
        normal_evidence_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
        normal_resources_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
    }
}
Require-LiteralText -Raw $gateRaw -Values @(
    'fresh `ACK_ACCEPTED` specifically for `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`',
    'fifteen non-handoff executor outputs',
    'does not require the not-yet-authored executor handoff',
    'writes its append-only handoff as output sixteen',
    'only permitted traversal is traversal performed internally by the exact declared `tools/validate_blueprint_tree.ps1` command'
) -Label 'Continuity gate sequencing and scope'

$leafRaw = Get-Content -Raw -LiteralPath $leafPath
$leafFront = Get-FrontMatterBlock -Raw $leafRaw
$expectedExecutorPaths = @(
    'docs/blueprints/ROOT_CAPSULE.md',
    'docs/blueprints/STATUS.md',
    'docs/blueprints/branches/resources/CAPSULE.md',
    'docs/blueprints/branches/resources/STATUS.md',
    'docs/blueprints/branches/evidence/BLUEPRINT.md',
    'docs/blueprints/branches/evidence/CAPSULE.md',
    'docs/blueprints/branches/evidence/STATUS.md',
    'docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md',
    'docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md',
    'docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md',
    'docs/blueprints/BLUEPRINT_INDEX.json',
    'docs/blueprints/TREE.md',
    'docs/blueprints/TRACEABILITY.md',
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md',
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-001.md'
)
$reviewerPath = 'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md'
if ($null -eq $leafFront) {
    Add-ValidationError 'Continuity leaf front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $leafFront -Label 'Continuity leaf' -Expected @{
        blueprint_id = 'ANG-WORK-CR0-CONTINUITY-001'
        parent_id = 'ANG-BP-ROOT'
        revision = '1'
        delivery_status = 'ready'
        activation_state = 'unusable_pending_revalidation'
        activation_revalidation = 'ANG-CR0-REVALIDATION-20260825-005'
        accountable_owner = 'ANG-BP-ROOT'
        human_authority = 'ANG-AUTH-PROJECT-OWNER-001'
        execution_owner = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001'
        executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        authorized_write_scope_owner = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001'
        independent_validator = 'ANG-AUTH-VALIDATOR-001'
        independent_gate_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001'
        independent_gate_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        independent_gate_write_scope_owner = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_gate_write_scope_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        gate_sha256 = $expectedGateHash
        rollback_ref = "ANG-BASELINE-CR0-CONTINUITY-001@sha256:$expectedBaselineHash"
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
        normal_evidence_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
        normal_resources_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
    }
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'authorized_write_scope') -Expected $expectedExecutorPaths -Label 'Executor authorized_write_scope'
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'executor_denied_write_scope') -Expected @($reviewerPath) -Label 'Executor denied scope'
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'independent_gate_write_scope') -Expected @($reviewerPath) -Label 'Reviewer write scope'
}

Require-LiteralText -Raw $leafRaw -Values @(
    'pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1',
    'pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1',
    '60 seconds per command',
    '600 seconds aggregate active time',
    '1 logical CPU',
    '512 MiB working set',
    '2 MiB total changed/new bytes',
    'fifteen non-handoff outputs',
    'Only after those three commands pass may the executor author its handoff as output sixteen',
    'The test must not require the not-yet-authored executor handoff',
    'sole traversal exception',
    'No schema, source, fixture, historical test, decision, receipt, handoff, baseline, ADR, policy, assessment, contract, normal gate, or threshold change'
) -Label 'Continuity leaf constraints'

if ($expectedExecutorPaths -contains $reviewerPath) {
    Add-ValidationError 'Executor and reviewer write scopes overlap'
}
$roleInstances = @(
    'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001',
    'ANG-AUTH-VALIDATOR-001',
    'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001',
    'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005',
    'ANG-BP-ROOT'
)
if (($roleInstances | Sort-Object -Unique).Count -ne $roleInstances.Count) {
    Add-ValidationError 'Continuity executor, validator, gate reviewer, revalidation reviewer, and recorder must be distinct'
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    Add-ValidationError 'Manifest v3 is missing'
}
if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    Add-ValidationError 'Revalidation-005 specification is missing'
}

$manifestRaw = if (Test-Path -LiteralPath $manifestPath) { Get-Content -Raw -LiteralPath $manifestPath } else { '' }
$manifestFront = Get-FrontMatterBlock -Raw $manifestRaw
$specRaw = if (Test-Path -LiteralPath $specPath) { Get-Content -Raw -LiteralPath $specPath } else { '' }
$specFront = Get-FrontMatterBlock -Raw $specRaw
$selfHash = Get-Sha256 -Path $selfPath
$manifestHash = Get-Sha256 -Path $manifestPath
$specHash = Get-Sha256 -Path $specPath

if ($null -eq $manifestFront) {
    Add-ValidationError 'Manifest v3 front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $manifestFront -Label 'Manifest v3' -Expected @{
        release_id = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
        version = '3'
        supersedes_version = '2'
        predecessor_manifest_v2_sha256 = '35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41'
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-005'
        revalidation_decision_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json'
        activation_base_commit = $expectedBaseCommit
        sole_ready_leaf = 'ANG-WORK-CR0-CONTINUITY-001@1'
        continuity_baseline_sha256 = $expectedBaselineHash
        continuity_gate_sha256 = $expectedGateHash
        continuity_leaf_sha256 = $expectedLeafHash
        continuity_executor = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001'
        continuity_executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        continuity_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001'
        continuity_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        continuity_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        continuity_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        continuity_result_recorder = 'ANG-BP-ROOT'
        revalidation_spec_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005.md'
        revalidation_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        revalidation_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005'
        revalidation_reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        revalidation_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        v3_validator_sha256 = $selfHash
        historical_v2_validator_sha256 = '50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94'
        formal_human_flourishing_gate_status = 'NOT_RUN'
        normal_evidence_schema_gate_status = 'NOT_RUN'
        normal_resource_design_gate_status = 'NOT_RUN'
        slice_status = 'NOT_PASSED'
        milestone_status = 'NOT_PASSED'
    }
}

if ($null -eq $specFront) {
    Add-ValidationError 'Revalidation-005 front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $specFront -Label 'Revalidation-005' -Expected @{
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-005'
        status = 'PENDING'
        release_manifest_version = '3'
        base_commit = $expectedBaseCommit
        pending_manifest_sha256 = (Get-ScalarField -FrontMatter $specFront -Name 'pending_manifest_sha256')
        validator_sha256 = $selfHash
        baseline_sha256 = $expectedBaselineHash
        gate_sha256 = $expectedGateHash
        leaf_sha256 = $expectedLeafHash
        reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005'
        reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        decision_writer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        decision_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json'
        allowed_dispositions = 'APPROVED|REJECTED|ESCALATE'
    }
}

$pendingManifestHash = if ($null -ne $specFront) { Get-ScalarField -FrontMatter $specFront -Name 'pending_manifest_sha256' } else { $null }
$status = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'status' } else { $null }
$revalidationStatus = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_status' } else { $null }
$decisionStatus = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_decision_status' } else { $null }
$decisionHashField = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_decision_sha256' } else { $null }
$pendingBanner = '> **PENDING / NON-AUTHORIZING.** This successor manifest grants no construction or execution authority. `ANG-WORK-CR0-CONTINUITY-001@1` is ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS.'
$pendingPhaseParagraph = 'Revalidation `ANG-CR0-REVALIDATION-20260825-005` is `PENDING`; its reserved decision path is absent. A separately authored decision with exact disposition `APPROVED`, followed by Root''s minimal `authorized` / `PASS` manifest transition and a successful authorized branch of the frozen v3 validator, is required before the one continuity leaf may start. Neither packet authoring nor PENDING static validation is permission.'
$authorizedBanner = '> **AUTHORIZED ONLY FOR `ANG-WORK-CR0-CONTINUITY-001@1`.** Revalidation 005 is independently `APPROVED`; Manifest v3 authorizes only the frozen continuity leaf after this validator passes in AUTHORIZED mode.'
$authorizedPhaseParagraph = 'Revalidation `ANG-CR0-REVALIDATION-20260825-005` is `PASS`; its separately authored decision is `APPROVED` and pinned in front matter. This transition grants no other authority and preserves every frozen scope, hash, denial, and non-equivalence state.'

$phase = $null
if ($status -eq 'pending_revalidation' -and $revalidationStatus -eq 'PENDING' -and $decisionStatus -eq 'ABSENT') {
    $phase = 'PENDING'
}
elseif ($status -eq 'authorized' -and $revalidationStatus -eq 'PASS' -and $decisionStatus -eq 'APPROVED') {
    $phase = 'AUTHORIZED'
}
else {
    Add-ValidationError "Invalid Manifest-v3 phase combination: status=$status revalidation_status=$revalidationStatus decision_status=$decisionStatus"
}

if ($phase -eq 'PENDING') {
    if (Test-Path -LiteralPath $decisionPath) {
        Add-ValidationError 'PENDING phase requires the revalidation-005 decision path to be absent'
    }
    if ($decisionHashField -ne 'ABSENT') {
        Add-ValidationError 'PENDING phase requires revalidation_decision_sha256: ABSENT'
    }
    if ($pendingManifestHash -ne $manifestHash) {
        Add-ValidationError "Revalidation spec does not pin the exact PENDING manifest: expected $manifestHash, found $pendingManifestHash"
    }
    Require-LiteralText -Raw $manifestRaw -Values @($pendingBanner, $pendingPhaseParagraph) -Label 'PENDING Manifest-v3 phase prose'
    if ($manifestRaw.Contains($authorizedBanner) -or $manifestRaw.Contains($authorizedPhaseParagraph)) {
        Add-ValidationError 'PENDING Manifest contains AUTHORIZED phase prose'
    }
}
elseif ($phase -eq 'AUTHORIZED') {
    if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) {
        Add-ValidationError 'AUTHORIZED phase requires the revalidation-005 decision file'
    }
    else {
        $actualDecisionHash = Get-Sha256 -Path $decisionPath
        if ($decisionHashField -ne $actualDecisionHash) {
            Add-ValidationError 'AUTHORIZED manifest does not pin the exact revalidation-005 decision hash'
        }
        try {
            $decision = Get-Content -Raw -LiteralPath $decisionPath | ConvertFrom-Json
            if ($decision.revalidation_id -ne 'ANG-CR0-REVALIDATION-20260825-005' -or $decision.disposition -ne 'APPROVED') {
                Add-ValidationError 'Revalidation-005 decision identity/disposition mismatch'
            }
            if ($decision.reviewer_role -ne 'ANG-AUTH-SAFETY-APPROVER-001' -or
                $decision.reviewer_instance -ne 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005' -or
                $decision.reviewer_session_ref -ne 'codex-subagent:/root/flourishing_red_team' -or
                $decision.reviewer_vocabulary_ack -ne 'ACK_ACCEPTED') {
                Add-ValidationError 'Revalidation-005 reviewer binding mismatch'
            }
            if ($decision.bindings.pending_manifest_sha256 -ne $pendingManifestHash -or
                $decision.bindings.spec_sha256 -ne $specHash -or
                $decision.bindings.validator_sha256 -ne $selfHash -or
                $decision.bindings.baseline_sha256 -ne $expectedBaselineHash -or
                $decision.bindings.gate_sha256 -ne $expectedGateHash -or
                $decision.bindings.leaf_sha256 -ne $expectedLeafHash -or
                $decision.bindings.base_commit -ne $expectedBaseCommit) {
                Add-ValidationError 'Revalidation-005 decision packet binding mismatch'
            }
            if ($decision.non_equivalence.normal_evidence_schema_gate -ne 'NOT_RUN' -or
                $decision.non_equivalence.normal_resource_design_gate -ne 'NOT_RUN' -or
                $decision.non_equivalence.human_flourishing_gate -ne 'NOT_RUN' -or
                $decision.non_equivalence.slice_00 -ne 'NOT_PASSED' -or
                $decision.non_equivalence.m0 -ne 'NOT_PASSED') {
                Add-ValidationError 'Revalidation-005 decision non-equivalence fields mismatch'
            }
            if ($decision.review_confirmation.continuity_leaf_executed -ne $false -or
                $decision.review_confirmation.continuity_test_executed -ne $false -or
                $decision.review_confirmation.continuity_outputs_created -ne $false -or
                $decision.review_confirmation.continuity_receipt_created -ne $false) {
                Add-ValidationError 'Revalidation-005 decision must confirm no continuity execution, test, output, or receipt during review'
            }
        }
        catch {
            Add-ValidationError "Unable to validate revalidation-005 decision: $($_.Exception.Message)"
        }
    }
    if ((Get-ScalarField -FrontMatter $manifestFront -Name 'pending_manifest_sha256') -ne $pendingManifestHash) {
        Add-ValidationError 'AUTHORIZED manifest must retain the exact PENDING manifest hash from the spec'
    }
    Require-LiteralText -Raw $manifestRaw -Values @($authorizedBanner, $authorizedPhaseParagraph) -Label 'AUTHORIZED Manifest-v3 phase prose'
    if ($manifestRaw.Contains($pendingBanner) -or $manifestRaw.Contains($pendingPhaseParagraph)) {
        Add-ValidationError 'AUTHORIZED Manifest retains the PENDING current-state prose'
    }
    $reconstructedPending = $manifestRaw
    $reconstructedPending = $reconstructedPending.Replace('status: authorized', 'status: pending_revalidation')
    $reconstructedPending = $reconstructedPending.Replace('revalidation_status: PASS', 'revalidation_status: PENDING')
    $reconstructedPending = $reconstructedPending.Replace('revalidation_decision_status: APPROVED', 'revalidation_decision_status: ABSENT')
    $reconstructedPending = $reconstructedPending.Replace("revalidation_decision_sha256: $decisionHashField", 'revalidation_decision_sha256: ABSENT')
    $reconstructedPending = $reconstructedPending.Replace("pending_manifest_sha256: $pendingManifestHash", 'pending_manifest_sha256: ABSENT_UNTIL_AUTHORIZED')
    $reconstructedPending = $reconstructedPending.Replace($authorizedBanner, $pendingBanner)
    $reconstructedPending = $reconstructedPending.Replace($authorizedPhaseParagraph, $pendingPhaseParagraph)
    $reconstructedPendingHash = Get-Utf8TextSha256 -Text $reconstructedPending
    if ($reconstructedPendingHash -ne $pendingManifestHash) {
        Add-ValidationError "AUTHORIZED Manifest is not the exact predeclared minimal transition from the frozen PENDING candidate: reconstructed $reconstructedPendingHash, expected $pendingManifestHash"
    }
}

if (Test-Path -LiteralPath $receiptPath) {
    Add-ValidationError 'Pre-start v3 validation requires the Continuity reviewer receipt to be absent'
}

Require-LiteralText -Raw $manifestRaw -Values @(
    'ANG-WORK-CR0-CONTINUITY-001@1',
    'ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS',
    'ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0',
    'D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B',
    'historical and non-repeatable',
    '117 unique role-owned outputs',
    'A later successor manifest is required before SAFETY or any other leaf can activate',
    'normal Evidence, Resources, and Human-Flourishing gates remain `NOT_RUN`',
    'Slice 00 and M0 remain `NOT_PASSED`',
    'no model/GPU/probe, network/package, recovered/real-person data, deployment, promotion, production, or external-use authority'
) -Label 'Manifest-v3 authority/non-equivalence'

$historicalManifestHashes = @(
    '35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41',
    '802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2',
    '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0',
    '5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289',
    'A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5',
    'F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F',
    '897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53',
    '9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893',
    '017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855',
    '1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9',
    '9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A',
    '50ED004747A4DABD9D306E8931D8BFD8A558F0CDE441596645A8E028F15CF9D2',
    '12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9',
    'D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B',
    '67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA',
    'A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2',
    'A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE',
    '916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A',
    '92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F',
    '7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2',
    'CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956',
    '162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9',
    'AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85',
    '92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A',
    '047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA',
    '50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94'
)
Require-LiteralText -Raw $manifestRaw -Values $historicalManifestHashes -Label 'Manifest-v3 immutable historical ledger'

Require-LiteralText -Raw $specRaw -Values @(
    'PENDING / NON-AUTHORIZING',
    'APPROVED|REJECTED|ESCALATE',
    'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005',
    'codex-subagent:/root/flourishing_red_team',
    'ACK_ACCEPTED',
    'pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1',
    'pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0-v3.ps1',
    'pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1',
    'two identical PENDING rounds',
    'do not execute the continuity leaf or test'
) -Label 'Revalidation-005 procedure'

if ($script:Errors.Count -gt 0) {
    foreach ($validationError in $script:Errors) {
        Write-Error $validationError -ErrorAction Continue
    }
    exit 1
}

if ($phase -eq 'PENDING') {
    Write-Output 'Construction Release 0 v3 validation PASS - PENDING / NON-AUTHORIZING continuity packet; 13 exact stale projections preserved, 4 future outputs absent, and no leaf execution authority exists.'
}
else {
    Write-Output 'Construction Release 0 v3 validation PASS - revalidation 005 independently APPROVED and Manifest v3 authorized only for ANG-WORK-CR0-CONTINUITY-001@1; all 17 targets remain at pre-start baseline.'
}
