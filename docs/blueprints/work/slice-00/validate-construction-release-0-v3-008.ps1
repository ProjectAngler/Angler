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

function Replace-ExactOnce {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][string]$OldValue,
        [Parameter(Mandatory)][string]$NewValue,
        [Parameter(Mandatory)][string]$Label
    )
    $count = [regex]::Matches($Text, [regex]::Escape($OldValue)).Count
    if ($count -ne 1) {
        Add-ValidationError "$Label expected exactly one source literal, found ${count}: $OldValue"
        return $Text
    }
    return $Text.Replace($OldValue, $NewValue)
}

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$releaseRoot = Join-Path $script:RepoRoot 'docs\blueprints\releases\construction-0'
$manifestPath = Join-Path $releaseRoot 'MANIFEST.md'
$baselinePath = Join-Path $releaseRoot 'baselines\ANG-BASELINE-CR0-CONTINUITY-004.json'
$gatePath = Join-Path $releaseRoot 'gates\ANG-GATE-CR0-CONTINUITY-004.md'
$leafPath = Join-Path $script:RepoRoot 'docs\blueprints\work\slice-00\ANG-WORK-CR0-CONTINUITY-004.md'
$specPath = Join-Path $releaseRoot 'revalidations\ANG-CR0-REVALIDATION-20260825-008.md'
$decisionPath = Join-Path $releaseRoot 'revalidations\ANG-CR0-REVALIDATION-20260825-008-decision.json'
$receiptPath = Join-Path $releaseRoot 'branch-receipts\CONTINUITY-004.md'
$rejectedReceiptPath = Join-Path $releaseRoot 'branch-receipts\CONTINUITY.md'
$rejectedAddendumPath = Join-Path $script:RepoRoot 'docs\blueprints\branches\evidence\children\evidence-schemas\handoffs\2026-08-25-cr0-scaffold-accepted.md'
$rejectedHandoffPath = Join-Path $script:RepoRoot 'docs\blueprints\work\slice-00\handoffs\ANG-WORK-CR0-CONTINUITY-002.md'
$rejectedTestPath = Join-Path $script:RepoRoot 'tests\synthetic\slice00\continuity\Test-Cr0ContinuationConsistency.ps1'
$failed007DecisionPath = Join-Path $releaseRoot 'revalidations\ANG-CR0-REVALIDATION-20260825-007-decision.json'
$selfPath = $MyInvocation.MyCommand.Path

$expectedBaseCommit = '21f7474ad40b46a6dc09ebab521f54c9089fbf50'
$expectedBaselineHash = '8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC'
$expectedGateHash = 'CAEB94B1C559E9D01A7836CB7D5DE55CB7E65D473F9C23A7E3C1CD464A1B56A6'
$expectedLeafHash = '6670C02625F6D0E841AA4A6ECF414641336221B8FD5D934123510D34987E8B88'

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
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-001.json' = 'EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7'
    'docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-001.md' = '85DAEB5E3FE72FE37EF40A141FB7DA8B2A133590C6520B2BC1BC55D0C94E20AC'
    'docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-001.md' = '8175422997DAF86D287A776E45BED2D7F48F3BA79D3EBC65D42130A222419116'
    'docs/blueprints/work/slice-00/validate-construction-release-0-v3.ps1' = 'EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005.md' = '31144831A0D194130BF2F068CE129D045BCD0D57038855EBEB92032C73019DB9'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json' = '467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3'
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-002.json' = '0BEF86FC8A56870E4B94BE1E057FD3C975D97D66DB1C4B775CC273AB373FFCE9'
    'docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-002.md' = '947A78F2E7EA9528B5B1997A0DA178E125BF0B0E05C7192B5547E31BFF7A2919'
    'docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-002.md' = '67873CE6B8CF51702357ED7D8473E023FEAEF1B9459313E1955719BB1F681680'
    'docs/blueprints/work/slice-00/validate-construction-release-0-v3-006.ps1' = '866AC1E579A4D38CA898B640AF2F2F8B464352BAB759ABC38F5D3C0A6D10D21A'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-006.md' = '5BDF2991AF0CE32949195C2479662E2F5592BAE3E860D9AEED674DB416E677C7'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-006-decision.json' = 'C34C11606057FC6F22B7428D4DCE9F707B7A64C1D8038FE9F356DC0CBED98296'
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md' = '8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8'
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-002.md' = '42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C'
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md' = 'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173'
    'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-003.json' = '134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28'
    'docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-003.md' = 'C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9'
    'docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-003.md' = 'D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD'
    'docs/blueprints/work/slice-00/validate-construction-release-0-v3-007.ps1' = 'A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B'
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007.md' = 'D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D'
    'tools/validate_blueprint_tree.ps1' = '421A06CCE6B5528F67ADF5869F1BB578C1F4B252785CEA96434BA7C6C4CD2BE7'
}

foreach ($entry in $immutableHashes.GetEnumerator()) {
    Assert-FileHash -RelativePath $entry.Key -ExpectedHash $entry.Value
}

Assert-FileHash -RelativePath 'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-004.json' -ExpectedHash $expectedBaselineHash
Assert-FileHash -RelativePath 'docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-004.md' -ExpectedHash $expectedGateHash
Assert-FileHash -RelativePath 'docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-004.md' -ExpectedHash $expectedLeafHash

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
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-004.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1' = @{ state = 'absent'; sha256 = $null; class = 'restore_or_remove' }
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-004.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md' = @{ state = 'absent'; sha256 = $null; class = 'preserve_on_failure' }
}

try {
    $baseline = Get-Content -Raw -LiteralPath $baselinePath | ConvertFrom-Json
    if ($baseline.baseline_id -ne 'ANG-BASELINE-CR0-CONTINUITY-004' -or $baseline.base_commit -ne $expectedBaseCommit -or [int]$baseline.leaf_revision -ne 1) {
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
    if ($baseline.rejected_predecessor.failed_packet_commit -ne 'c98fbe85ceebb7bddd167b33b5a7459ce54110bc' -or
        $baseline.rejected_predecessor.receipt_sha256 -ne 'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173' -or
        $baseline.rejected_predecessor.addendum_sha256 -ne '8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8' -or
        $baseline.rejected_predecessor.handoff_sha256 -ne '42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C' -or
        $baseline.rejected_predecessor.test_historical_sha256 -ne '1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC' -or
        $baseline.rejected_predecessor.test_state -ne 'absent') {
        Add-ValidationError 'Continuity baseline rejected-predecessor binding mismatch'
    }
    if ($baseline.failed_pre_activation_packet_007.commit -ne $expectedBaseCommit -or
        $baseline.failed_pre_activation_packet_007.manifest_sha256 -ne '7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33' -or
        $baseline.failed_pre_activation_packet_007.baseline_sha256 -ne '134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28' -or
        $baseline.failed_pre_activation_packet_007.gate_sha256 -ne 'C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9' -or
        $baseline.failed_pre_activation_packet_007.leaf_sha256 -ne 'D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD' -or
        $baseline.failed_pre_activation_packet_007.validator_sha256 -ne 'A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B' -or
        $baseline.failed_pre_activation_packet_007.spec_sha256 -ne 'D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D' -or
        $baseline.failed_pre_activation_packet_007.failure_class -ne 'AUDIT_FAILED_PRE_ACTIVATION_NO_AUTHORITY' -or
        $baseline.failed_pre_activation_packet_007.decision_state -ne 'absent' -or
        $baseline.failed_pre_activation_packet_007.leaf_execution_state -ne 'not_run') {
        Add-ValidationError 'Continuity baseline packet-007 failure binding mismatch'
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

if (Test-Path -LiteralPath $rejectedTestPath) {
    Add-ValidationError 'Rejected continuity-002 test path must remain absent and must not be reused'
}
if (Test-Path -LiteralPath $failed007DecisionPath) {
    Add-ValidationError 'Audit-failed packet-007 decision path must remain absent'
}
$failed007FuturePaths = @(
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-003.md',
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-003.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-003.md',
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-003.md'
)
foreach ($relative in $failed007FuturePaths) {
    if (Test-Path -LiteralPath (Join-Path $script:RepoRoot $relative)) {
        Add-ValidationError "Audit-failed packet-007 future path must remain absent: $relative"
    }
}
$rejectedReceiptRaw = Get-Content -Raw -LiteralPath $rejectedReceiptPath
Require-LiteralText -Raw $rejectedReceiptRaw -Values @(
    'disposition: RECONCILIATION_REJECTED',
    '8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8',
    '42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C',
    '1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC',
    'six',
    '33'
) -Label 'Immutable continuity-002 rejection receipt'

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
    if ($evidenceDecision.disposition -ne 'SCAFFOLD_ACCEPTED' -or
        $evidenceDecision.gate.spec_sha256 -ne 'A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5' -or
        $evidenceDecision.bindings.leaf.sha256 -ne '5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289' -or
        $evidenceDecision.bindings.rollback_baseline.sha256 -ne 'F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F' -or
        $evidenceDecision.bindings.release_manifest.sha256 -ne '802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2') {
        Add-ValidationError 'Evidence decision disposition or leaf/gate/baseline/execution-Manifest binding changed'
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
    '`ANG-WORK-CR0-RESOURCES-001@3`; SHA-256 `67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA`',
    '`ANG-GATE-CR0-RESOURCES-001@2`; SHA-256 `A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2`',
    '`ANG-BASELINE-CR0-RESOURCES-002`; SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`',
    'SHA-256 `35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41`',
    'SHA-256 `12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9`',
    'SHA-256 `916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A`',
    '92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F',
    '7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2',
    'CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956',
    '162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9',
    'AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85',
    '92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A',
    '047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA',
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
        gate_id = 'ANG-GATE-CR0-CONTINUITY-004'
        version = '1'
        status = 'specified'
        activation_state = 'unusable_pending_revalidation'
        activation_revalidation = 'ANG-CR0-REVALIDATION-20260825-008'
        release_manifest_version = '3'
        leaf = 'ANG-WORK-CR0-CONTINUITY-004@1'
        baseline_sha256 = $expectedBaselineHash
        executor = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004'
        executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        independent_verifier = 'ANG-AUTH-VALIDATOR-001'
        independent_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004'
        independent_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        allowed_dispositions = 'RECONCILIATION_ACCEPTED|RECONCILIATION_REJECTED|ESCALATE'
        decision_path = 'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md'
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
        normal_evidence_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
        normal_resources_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
    }
}
$liveBaselinePattern = '(?m)^- `ANG-BASELINE-CR0-CONTINUITY-004` SHA-256 `(?<hash>[A-F0-9]{64})` is reverified immediately before the first write:'
$liveBaselineMatches = [regex]::Matches($gateRaw, $liveBaselinePattern)
if ($liveBaselineMatches.Count -ne 1) {
    Add-ValidationError "Gate must contain exactly one canonical live continuity-004 baseline binding; found $($liveBaselineMatches.Count)"
}
elseif ($liveBaselineMatches[0].Groups['hash'].Value -ne $expectedBaselineHash) {
    Add-ValidationError "Gate live continuity-004 baseline binding is stale or competing: $($liveBaselineMatches[0].Groups['hash'].Value)"
}
$allInlineBaselineBindings = [regex]::Matches($gateRaw, '(?m)`ANG-BASELINE-CR0-CONTINUITY-004`[^\r\n]{0,96}?`(?<hash>[A-F0-9]{64})`')
if ($allInlineBaselineBindings.Count -lt 1) {
    Add-ValidationError 'Gate body lacks an inline continuity-004 baseline binding'
}
foreach ($binding in $allInlineBaselineBindings) {
    if ($binding.Groups['hash'].Value -ne $expectedBaselineHash) {
        Add-ValidationError "Gate body has competing continuity-004 baseline hash: $($binding.Groups['hash'].Value)"
    }
}
$stale007Binding = '`ANG-BASELINE-CR0-CONTINUITY-004` SHA-256 `FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D`'
if ($gateRaw.Contains($stale007Binding)) {
    Add-ValidationError 'Gate body reuses the stale packet-007 baseline binding'
}
Require-LiteralText -Raw $gateRaw -Values @(
    'fresh `ACK_ACCEPTED` specifically for `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`',
    '## Required child gates',
    'fifteen non-handoff executor outputs',
    'does not require the not-yet-authored executor handoff',
    'writes its append-only handoff as output sixteen',
    'exactly `64` named cases',
    'reruns the same consolidated 64-case test measurement command exactly once',
    'ANG-METRICS',
    '`ANG-BASELINE-CR0-CONTINUITY-004` SHA-256 `8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC` is reverified immediately before the first write:',
    'Packet 007 is immutable audit-failed pre-activation evidence',
    'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173',
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
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-004.md',
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-004.md'
)
$reviewerPath = 'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md'
$expectedDeniedPaths = @(
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md',
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-002.md',
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md',
    'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-003.md',
    'tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-003.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-003.md',
    'docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-003.md',
    $reviewerPath
)
if ($null -eq $leafFront) {
    Add-ValidationError 'Continuity leaf front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $leafFront -Label 'Continuity leaf' -Expected @{
        blueprint_id = 'ANG-WORK-CR0-CONTINUITY-004'
        parent_id = 'ANG-BP-ROOT'
        revision = '1'
        delivery_status = 'ready'
        activation_state = 'unusable_pending_revalidation'
        activation_revalidation = 'ANG-CR0-REVALIDATION-20260825-008'
        activation_base_commit = $expectedBaseCommit
        accountable_owner = 'ANG-BP-ROOT'
        human_authority = 'ANG-AUTH-PROJECT-OWNER-001'
        execution_owner = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004'
        executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        authorized_write_scope_owner = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004'
        independent_validator = 'ANG-AUTH-VALIDATOR-001'
        independent_gate_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004'
        independent_gate_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        independent_gate_write_scope_owner = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_gate_write_scope_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        gate_sha256 = $expectedGateHash
        rollback_ref = "ANG-BASELINE-CR0-CONTINUITY-004@sha256:$expectedBaselineHash"
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
        normal_evidence_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
        normal_resources_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
    }
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'authorized_write_scope') -Expected $expectedExecutorPaths -Label 'Executor authorized_write_scope'
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'executor_denied_write_scope') -Expected $expectedDeniedPaths -Label 'Executor denied scope'
    Assert-ExactSet -Actual @(Get-ListField -FrontMatter $leafFront -Name 'independent_gate_write_scope') -Expected @($reviewerPath) -Label 'Reviewer write scope'
}

Require-LiteralText -Raw $leafRaw -Values @(
    '$scriptPath="tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1"',
    '$scriptPath="tools/validate_blueprint_tree.ps1"',
    '$p.ProcessorAffinity=[IntPtr]1',
    'ANG-METRICS',
    '60 seconds per command',
    '600 seconds aggregate active time',
    '1 logical CPU',
    '512 MiB peak working set',
    '2 MiB total changed/new bytes',
    'exactly 64 named cases',
    'all 17 pre-start target identities',
    'all 16 post-write executor-output identities',
    'FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D',
    'packet-007 hashes/failure class',
    'authoritative baseline-004 body binding',
    'fifteen non-handoff outputs',
    'Only after those two commands pass may the executor author its handoff as output sixteen',
    'The test must not require the not-yet-authored executor handoff',
    'sole traversal exception',
    'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173',
    'No schema, source, fixture, historical test, decision, receipt, handoff, baseline, ADR, policy, assessment, contract, normal gate, or threshold change'
) -Label 'Continuity leaf constraints'

if ($expectedExecutorPaths -contains $reviewerPath) {
    Add-ValidationError 'Executor and reviewer write scopes overlap'
}
$roleInstances = @(
    'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004',
    'ANG-AUTH-VALIDATOR-001',
    'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004',
    'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008',
    'ANG-BP-ROOT'
)
if (($roleInstances | Sort-Object -Unique).Count -ne $roleInstances.Count) {
    Add-ValidationError 'Continuity executor, validator, gate reviewer, revalidation reviewer, and recorder must be distinct'
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    Add-ValidationError 'Manifest v3 is missing'
}
if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    Add-ValidationError 'Revalidation-008 specification is missing'
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
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-008'
        revalidation_decision_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-008-decision.json'
        activation_base_commit = $expectedBaseCommit
        sole_ready_leaf = 'ANG-WORK-CR0-CONTINUITY-004@1'
        continuity_baseline_sha256 = $expectedBaselineHash
        continuity_gate_sha256 = $expectedGateHash
        continuity_leaf_sha256 = $expectedLeafHash
        continuity_executor = 'ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004'
        continuity_executor_task_id = '01a03a80-bb20-7d01-acf6-f50ca4856be5'
        continuity_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004'
        continuity_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        continuity_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        continuity_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        continuity_result_recorder = 'ANG-BP-ROOT'
        revalidation_spec_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-008.md'
        revalidation_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        revalidation_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008'
        revalidation_reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        revalidation_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        v3_validator_sha256 = $selfHash
        historical_v2_validator_sha256 = '50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94'
        historical_005_reviewed_pending_manifest_sha256 = '08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F'
        historical_005_decision_sha256 = '467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3'
        historical_005_attempted_authorized_manifest_sha256 = '9578BD1B97451636449254CF1496387DA9602240B51401CC85595237E657C3E5'
        historical_005_validator_sha256 = 'EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D'
        historical_005_preserved_commit = '84ff08b484197755c3fed66e7dc06988539e456e'
        historical_005_authorization_status = 'FAILED_VALIDATION_NO_AUTHORITY'
        historical_006_pending_manifest_sha256 = 'E41000C14FE7BA0F88FED46DA29DBCFAF3D3DD15D2DB54E001A34A76CFCBCE02'
        historical_006_authorized_manifest_sha256 = 'D10284F3A81B85DFBF1F342EFF213C5B2E67CABDB229552C6F11165C896159D7'
        historical_006_decision_sha256 = 'C34C11606057FC6F22B7428D4DCE9F707B7A64C1D8038FE9F356DC0CBED98296'
        historical_006_validator_sha256 = '866AC1E579A4D38CA898B640AF2F2F8B464352BAB759ABC38F5D3C0A6D10D21A'
        historical_continuity_002_failed_packet_commit = 'c98fbe85ceebb7bddd167b33b5a7459ce54110bc'
        historical_continuity_002_receipt_sha256 = 'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173'
        historical_continuity_002_addendum_sha256 = '8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8'
        historical_continuity_002_handoff_sha256 = '42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C'
        historical_continuity_002_test_sha256 = '1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC'
        historical_continuity_002_disposition = 'RECONCILIATION_REJECTED'
        historical_007_commit = '21f7474ad40b46a6dc09ebab521f54c9089fbf50'
        historical_007_manifest_sha256 = '7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33'
        historical_007_baseline_sha256 = '134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28'
        historical_007_gate_sha256 = 'C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9'
        historical_007_leaf_sha256 = 'D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD'
        historical_007_validator_sha256 = 'A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B'
        historical_007_spec_sha256 = 'D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D'
        historical_007_failure_status = 'AUDIT_FAILED_PRE_ACTIVATION_NO_AUTHORITY'
        historical_007_decision_status = 'ABSENT'
        historical_007_leaf_execution_status = 'NOT_RUN'
        formal_human_flourishing_gate_status = 'NOT_RUN'
        normal_evidence_schema_gate_status = 'NOT_RUN'
        normal_resource_design_gate_status = 'NOT_RUN'
        slice_status = 'NOT_PASSED'
        milestone_status = 'NOT_PASSED'
    }
}

if ($null -eq $specFront) {
    Add-ValidationError 'Revalidation-008 front matter is missing'
}
else {
    Assert-ScalarFields -FrontMatter $specFront -Label 'Revalidation-008' -Expected @{
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-008'
        status = 'PENDING'
        release_manifest_version = '3'
        base_commit = $expectedBaseCommit
        pending_manifest_sha256 = (Get-ScalarField -FrontMatter $specFront -Name 'pending_manifest_sha256')
        validator_sha256 = $selfHash
        baseline_sha256 = $expectedBaselineHash
        gate_sha256 = $expectedGateHash
        leaf_sha256 = $expectedLeafHash
        reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008'
        reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        decision_writer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        decision_path = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-008-decision.json'
        allowed_dispositions = 'APPROVED|REJECTED|ESCALATE'
        failed_predecessor_revalidation_id = 'ANG-CR0-REVALIDATION-20260825-007'
        failed_predecessor_manifest_sha256 = '7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33'
        failed_predecessor_status = 'AUDIT_FAILED_PRE_ACTIVATION_NO_AUTHORITY'
        failed_predecessor_decision_state = 'ABSENT'
        rejected_predecessor_leaf = 'ANG-WORK-CR0-CONTINUITY-002@1'
        rejected_predecessor_receipt_sha256 = 'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173'
        rejected_predecessor_status = 'RECONCILIATION_REJECTED'
    }
}

$pendingManifestHash = if ($null -ne $specFront) { Get-ScalarField -FrontMatter $specFront -Name 'pending_manifest_sha256' } else { $null }
$status = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'status' } else { $null }
$revalidationStatus = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_status' } else { $null }
$decisionStatus = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_decision_status' } else { $null }
$decisionHashField = if ($null -ne $manifestFront) { Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_decision_sha256' } else { $null }
$pendingPhaseLine = '> **PENDING / NON-AUTHORIZING.** Revalidation 008 has no decision; `ANG-WORK-CR0-CONTINUITY-004@1` has no execution authority.'
$authorizedPhaseLine = '> **AUTHORIZED ONLY FOR `ANG-WORK-CR0-CONTINUITY-004@1`.** Revalidation 008 is independently `APPROVED`; authority exists only after this validator returns AUTHORIZED PASS.'
$sharedAuthorityLine = 'This successor packet covers only the frozen continuity reconciliation; every other leaf remains unauthorized in both phases.'

function Assert-DecisionObject {
    param(
        [Parameter(Mandatory)]$Decision,
        [Parameter(Mandatory)][string]$Label
    )
    if ($Decision.revalidation_id -ne 'ANG-CR0-REVALIDATION-20260825-008' -or $Decision.disposition -ne 'APPROVED') {
        Add-ValidationError "$Label decision identity/disposition mismatch"
    }
    if ($Decision.reviewer_role -ne 'ANG-AUTH-SAFETY-APPROVER-001' -or
        $Decision.reviewer_instance -ne 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008' -or
        $Decision.reviewer_session_ref -ne 'codex-subagent:/root/flourishing_red_team' -or
        $Decision.reviewer_vocabulary_ack -ne 'ACK_ACCEPTED') {
        Add-ValidationError "$Label reviewer binding mismatch"
    }
    if ($Decision.bindings.pending_manifest_sha256 -ne $pendingManifestHash -or
        $Decision.bindings.spec_sha256 -ne $specHash -or
        $Decision.bindings.validator_sha256 -ne $selfHash -or
        $Decision.bindings.baseline_sha256 -ne $expectedBaselineHash -or
        $Decision.bindings.gate_sha256 -ne $expectedGateHash -or
        $Decision.bindings.leaf_sha256 -ne $expectedLeafHash -or
        $Decision.bindings.base_commit -ne $expectedBaseCommit) {
        Add-ValidationError "$Label packet binding mismatch"
    }
    if ($Decision.non_equivalence.normal_evidence_schema_gate -ne 'NOT_RUN' -or
        $Decision.non_equivalence.normal_resource_design_gate -ne 'NOT_RUN' -or
        $Decision.non_equivalence.human_flourishing_gate -ne 'NOT_RUN' -or
        $Decision.non_equivalence.slice_00 -ne 'NOT_PASSED' -or
        $Decision.non_equivalence.m0 -ne 'NOT_PASSED') {
        Add-ValidationError "$Label non-equivalence fields mismatch"
    }
    if ($Decision.review_confirmation.continuity_leaf_executed -ne $false -or
        $Decision.review_confirmation.continuity_test_executed -ne $false -or
        $Decision.review_confirmation.continuity_outputs_created -ne $false -or
        $Decision.review_confirmation.continuity_receipt_created -ne $false) {
        Add-ValidationError "$Label must confirm no continuity execution, test, output, or receipt during review"
    }
}

function Convert-ToAuthorizedManifest {
    param(
        [Parameter(Mandatory)][string]$PendingRaw,
        [Parameter(Mandatory)][string]$DecisionHash,
        [Parameter(Mandatory)][string]$PendingHash,
        [Parameter(Mandatory)][string]$Label
    )
    $result = $PendingRaw
    $result = Replace-ExactOnce -Text $result -OldValue 'status: pending_revalidation' -NewValue 'status: authorized' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'revalidation_status: PENDING' -NewValue 'revalidation_status: PASS' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'revalidation_decision_status: ABSENT' -NewValue 'revalidation_decision_status: APPROVED' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'revalidation_decision_sha256: ABSENT' -NewValue "revalidation_decision_sha256: $DecisionHash" -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'pending_manifest_sha256: ABSENT_UNTIL_AUTHORIZED' -NewValue "pending_manifest_sha256: $PendingHash" -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue $pendingPhaseLine -NewValue $authorizedPhaseLine -Label $Label
    return $result
}

function Convert-ToPendingManifest {
    param(
        [Parameter(Mandatory)][string]$AuthorizedRaw,
        [Parameter(Mandatory)][string]$DecisionHash,
        [Parameter(Mandatory)][string]$PendingHash,
        [Parameter(Mandatory)][string]$Label
    )
    $result = $AuthorizedRaw
    $result = Replace-ExactOnce -Text $result -OldValue 'status: authorized' -NewValue 'status: pending_revalidation' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'revalidation_status: PASS' -NewValue 'revalidation_status: PENDING' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue 'revalidation_decision_status: APPROVED' -NewValue 'revalidation_decision_status: ABSENT' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue "revalidation_decision_sha256: $DecisionHash" -NewValue 'revalidation_decision_sha256: ABSENT' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue "pending_manifest_sha256: $PendingHash" -NewValue 'pending_manifest_sha256: ABSENT_UNTIL_AUTHORIZED' -Label $Label
    $result = Replace-ExactOnce -Text $result -OldValue $authorizedPhaseLine -NewValue $pendingPhaseLine -Label $Label
    return $result
}

function Assert-AuthorizedStaticState {
    param(
        [Parameter(Mandatory)][string]$AuthorizedRaw,
        [Parameter(Mandatory)]$Decision,
        [Parameter(Mandatory)][string]$DecisionHash,
        [Parameter(Mandatory)][string]$Label
    )
    $authorizedFront = Get-FrontMatterBlock -Raw $AuthorizedRaw
    if ($null -eq $authorizedFront) {
        Add-ValidationError "$Label authorized Manifest front matter is missing"
        return
    }
    Assert-ScalarFields -FrontMatter $authorizedFront -Label "$Label authorized Manifest" -Expected @{
        status = 'authorized'
        revalidation_status = 'PASS'
        revalidation_decision_status = 'APPROVED'
        revalidation_decision_sha256 = $DecisionHash
        pending_manifest_sha256 = $pendingManifestHash
    }
    Require-LiteralText -Raw $AuthorizedRaw -Values @($authorizedPhaseLine, $sharedAuthorityLine) -Label "$Label AUTHORIZED phase prose"
    if ($AuthorizedRaw.Contains($pendingPhaseLine)) {
        Add-ValidationError "$Label AUTHORIZED Manifest retains the PENDING phase line"
    }
    Assert-DecisionObject -Decision $Decision -Label "$Label revalidation-008 decision"
    $reconstructedPending = Convert-ToPendingManifest -AuthorizedRaw $AuthorizedRaw -DecisionHash $DecisionHash -PendingHash $pendingManifestHash -Label "$Label reverse transition"
    $reconstructedPendingHash = Get-Utf8TextSha256 -Text $reconstructedPending
    if ($reconstructedPendingHash -ne $pendingManifestHash) {
        Add-ValidationError "$Label AUTHORIZED Manifest is not the exact six-edit transition from the frozen PENDING candidate: reconstructed $reconstructedPendingHash, expected $pendingManifestHash"
    }
}

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
        Add-ValidationError 'PENDING phase requires the revalidation-008 decision path to be absent'
    }
    if ($decisionHashField -ne 'ABSENT') {
        Add-ValidationError 'PENDING phase requires revalidation_decision_sha256: ABSENT'
    }
    if ((Get-ScalarField -FrontMatter $manifestFront -Name 'pending_manifest_sha256') -ne 'ABSENT_UNTIL_AUTHORIZED') {
        Add-ValidationError 'PENDING phase requires pending_manifest_sha256: ABSENT_UNTIL_AUTHORIZED'
    }
    if ($pendingManifestHash -ne $manifestHash) {
        Add-ValidationError "Revalidation spec does not pin the exact PENDING manifest: expected $manifestHash, found $pendingManifestHash"
    }
    Require-LiteralText -Raw $manifestRaw -Values @($pendingPhaseLine, $sharedAuthorityLine) -Label 'PENDING Manifest-v3 phase prose'
    if ($manifestRaw.Contains($authorizedPhaseLine)) {
        Add-ValidationError 'PENDING Manifest contains AUTHORIZED phase prose'
    }
    $simulatedDecisionJson = [ordered]@{
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-008'
        disposition = 'APPROVED'
        reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008'
        reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        bindings = [ordered]@{
            pending_manifest_sha256 = $pendingManifestHash
            spec_sha256 = $specHash
            validator_sha256 = $selfHash
            baseline_sha256 = $expectedBaselineHash
            gate_sha256 = $expectedGateHash
            leaf_sha256 = $expectedLeafHash
            base_commit = $expectedBaseCommit
        }
        non_equivalence = [ordered]@{
            normal_evidence_schema_gate = 'NOT_RUN'
            normal_resource_design_gate = 'NOT_RUN'
            human_flourishing_gate = 'NOT_RUN'
            slice_00 = 'NOT_PASSED'
            m0 = 'NOT_PASSED'
        }
        review_confirmation = [ordered]@{
            continuity_leaf_executed = $false
            continuity_test_executed = $false
            continuity_outputs_created = $false
            continuity_receipt_created = $false
        }
    } | ConvertTo-Json -Depth 6 -Compress
    $simulatedDecisionHash = Get-Utf8TextSha256 -Text $simulatedDecisionJson
    try {
        $simulatedDecision = $simulatedDecisionJson | ConvertFrom-Json
        $selfTestErrorCount = $script:Errors.Count
        $simulatedAuthorized = Convert-ToAuthorizedManifest -PendingRaw $manifestRaw -DecisionHash $simulatedDecisionHash -PendingHash $pendingManifestHash -Label 'AUTHORIZED transition self-test forward transition'
        Assert-AuthorizedStaticState -AuthorizedRaw $simulatedAuthorized -Decision $simulatedDecision -DecisionHash $simulatedDecisionHash -Label 'AUTHORIZED transition self-test'
        $script:AuthorizedTransitionSelfTestPassed = ($script:Errors.Count -eq $selfTestErrorCount)
    }
    catch {
        $script:AuthorizedTransitionSelfTestPassed = $false
        Add-ValidationError "AUTHORIZED transition self-test could not parse its in-memory decision: $($_.Exception.Message)"
    }
    if (-not $script:AuthorizedTransitionSelfTestPassed) {
        Add-ValidationError 'AUTHORIZED transition self-test did not pass'
    }
}
elseif ($phase -eq 'AUTHORIZED') {
    if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) {
        Add-ValidationError 'AUTHORIZED phase requires the revalidation-008 decision file'
    }
    else {
        $actualDecisionHash = Get-Sha256 -Path $decisionPath
        if ($decisionHashField -ne $actualDecisionHash) {
            Add-ValidationError 'AUTHORIZED manifest does not pin the exact revalidation-008 decision hash'
        }
        try {
            $decision = Get-Content -Raw -LiteralPath $decisionPath | ConvertFrom-Json
            Assert-AuthorizedStaticState -AuthorizedRaw $manifestRaw -Decision $decision -DecisionHash $actualDecisionHash -Label 'Real AUTHORIZED branch'
        }
        catch {
            Add-ValidationError "Unable to validate revalidation-008 decision: $($_.Exception.Message)"
        }
    }
}

if (Test-Path -LiteralPath $receiptPath) {
    Add-ValidationError 'Pre-start v3 validation requires the Continuity reviewer receipt to be absent'
}

Require-LiteralText -Raw $manifestRaw -Values @(
    'ANG-WORK-CR0-CONTINUITY-004@1',
    'ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0',
    'D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B',
    'historical and non-repeatable',
    'The active 008 role partition contains exactly sixteen executor outputs and one fresh reviewer receipt',
    'Continuity-002 is immutable `RECONCILIATION_REJECTED` history',
    'Packet 007 is immutable audit-failed pre-activation history',
    'FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D',
    'authoritative baseline-004 gate-body binding',
    'all 14/14 Evidence implementation identities',
    'complete Resources binding/output map',
    'exactly 64 named cases',
    'single consolidated PENDING packet-validation round',
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
    '50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94',
    '08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F',
    '467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3',
    '9578BD1B97451636449254CF1496387DA9602240B51401CC85595237E657C3E5',
    'EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D',
    '84ff08b484197755c3fed66e7dc06988539e456e'
    'E41000C14FE7BA0F88FED46DA29DBCFAF3D3DD15D2DB54E001A34A76CFCBCE02'
    'D10284F3A81B85DFBF1F342EFF213C5B2E67CABDB229552C6F11165C896159D7'
    'C34C11606057FC6F22B7428D4DCE9F707B7A64C1D8038FE9F356DC0CBED98296'
    '866AC1E579A4D38CA898B640AF2F2F8B464352BAB759ABC38F5D3C0A6D10D21A'
    'c98fbe85ceebb7bddd167b33b5a7459ce54110bc'
    'BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173'
    '8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8'
    '42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C'
    '1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC'
    '21f7474ad40b46a6dc09ebab521f54c9089fbf50'
    '7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33'
    '134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28'
    'C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9'
    'D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD'
    'A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B'
    'D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D'
    'FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D'
)
Require-LiteralText -Raw $manifestRaw -Values $historicalManifestHashes -Label 'Manifest-v3 immutable historical ledger'
Require-LiteralText -Raw $manifestRaw -Values @(
    'exit 1',
    'Write-Error: Manifest-v3 authority/non-equivalence is missing required literal: ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS',
    '005 grants no execution authority because authorized validation failed',
    'no leaf, test, output, or receipt ran'
) -Label 'Manifest-v3 immutable revalidation-005 failure history'

Require-LiteralText -Raw $specRaw -Values @(
    'PENDING / NON-AUTHORIZING',
    'APPROVED|REJECTED|ESCALATE',
    'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-008',
    'codex-subagent:/root/flourishing_red_team',
    'ACK_ACCEPTED',
    'pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1',
    'pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0-v3-008.ps1',
    'pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1',
    'one consolidated PENDING validation round',
    'AUTHORIZED_TRANSITION_SELF_TEST PASS',
    'exactly six Manifest edits',
    'Continuity-002 is immutable rejected evidence',
    'Packet 007 is immutable audit-failed evidence',
    'FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D',
    'canonical live continuity-004 baseline binding',
    'Do not execute the continuity leaf or test'
) -Label 'Revalidation-008 procedure'

if ($script:Errors.Count -gt 0) {
    foreach ($validationError in $script:Errors) {
        Write-Error $validationError -ErrorAction Continue
    }
    exit 1
}

if ($phase -eq 'PENDING') {
    Write-Output 'Construction Release 0 v3-008 validation PASS - PENDING / NON-AUTHORIZING; AUTHORIZED_TRANSITION_SELF_TEST PASS; 13 exact projections preserved, 4 future outputs absent, and no leaf execution authority exists.'
}
else {
    Write-Output 'Construction Release 0 v3-008 validation PASS - revalidation 008 independently APPROVED and Manifest v3 authorized only for ANG-WORK-CR0-CONTINUITY-004@1; all 17 targets remain at pre-start baseline.'
}
