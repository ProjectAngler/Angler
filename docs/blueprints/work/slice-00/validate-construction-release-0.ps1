[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$blueprintRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot '..\..'))
$releaseRoot = Join-Path $blueprintRoot 'releases\construction-0'
$errors = [System.Collections.Generic.List[string]]::new()

function Add-ReleaseError {
    param([Parameter(Mandatory)][string]$Message)
    $errors.Add($Message)
}

function Get-FrontMatter {
    param([Parameter(Mandatory)][string]$Path)
    $raw = Get-Content -Raw -LiteralPath $Path
    $match = [regex]::Match($raw, '\A---\s*\r?\n(?<front>.*?)\r?\n---\s*\r?\n', 'Singleline')
    if (-not $match.Success) {
        Add-ReleaseError "Missing front matter: $Path"
        return ''
    }
    return $match.Groups['front'].Value
}

function Get-ScalarField {
    param(
        [Parameter(Mandatory)][string]$FrontMatter,
        [Parameter(Mandatory)][string]$Name
    )
    $match = [regex]::Match($FrontMatter, "(?m)^$([regex]::Escape($Name)):\s*(?<value>[^\r\n]+)\s*$")
    if ($match.Success) { return $match.Groups['value'].Value.Trim() }
    return $null
}

$leafSpecs = @(
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-SAFETY-001'; File = 'ANG-WORK-CR0-SAFETY-001.md'; Owner = 'ANG-BP-SAFETY'; Status = 'not_ready'; Test = 'tests/synthetic/slice00/safety/Test-Cr0SafetyReleaseReference.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-RESOURCES-001'; File = 'ANG-WORK-CR0-RESOURCES-001.md'; Owner = 'ANG-BP-RESOURCES'; Status = 'ready'; Test = 'tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-WORLDS-001'; File = 'ANG-WORK-CR0-WORLDS-001.md'; Owner = 'ANG-BP-WORLDS'; Status = 'blocked'; Test = 'tests/synthetic/slice00/worlds/Test-Cr0WorldScaffold.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-RUNTIME-001'; File = 'ANG-WORK-CR0-RUNTIME-001.md'; Owner = 'ANG-BP-RUNTIME'; Status = 'not_ready'; Test = 'tests/synthetic/slice00/runtime/Test-Cr0RuntimeScaffold.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-SCIENCE-001'; File = 'ANG-WORK-CR0-SCIENCE-001.md'; Owner = 'ANG-BP-SCIENCE'; Status = 'blocked'; Test = 'tests/synthetic/slice00/science/Test-Cr0ScienceScaffold.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-LEARNING-001'; File = 'ANG-WORK-CR0-LEARNING-001.md'; Owner = 'ANG-BP-LEARNING'; Status = 'not_ready'; Test = 'tests/synthetic/slice00/learning/Test-Cr0LearningScaffold.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-TOOLS-001'; File = 'ANG-WORK-CR0-TOOLS-001.md'; Owner = 'ANG-BP-TOOLS'; Status = 'not_ready'; Test = 'tests/synthetic/slice00/tools/Test-Cr0ToolsDeferral.ps1' },
    [pscustomobject]@{ Id = 'ANG-WORK-CR0-INTEGRATION-001'; File = 'ANG-WORK-CR0-INTEGRATION-001.md'; Owner = 'ANG-BP-ROOT'; Status = 'blocked'; Test = 'tests/synthetic/slice00/integration/Test-Cr0Integration.ps1' }
)

$requiredReleaseFiles = @(
    (Join-Path $releaseRoot 'MANIFEST.md'),
    (Join-Path $blueprintRoot 'branches\safety\policies\ANG-POL-LOCAL-SCAFFOLD-001.md'),
    (Join-Path $blueprintRoot 'branches\safety\assessments\ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md'),
    (Join-Path $blueprintRoot 'branches\safety\gates\ANG-GATE-CONSTRUCTION-RELEASE-0-001.md')
)
foreach ($requiredFile in $requiredReleaseFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Add-ReleaseError "Missing release artifact: $requiredFile"
    }
}

$manifestPath = Join-Path $releaseRoot 'MANIFEST.md'
$manifestRaw = if (Test-Path -LiteralPath $manifestPath) { Get-Content -Raw -LiteralPath $manifestPath } else { '' }
$manifestFront = if ($manifestRaw) { Get-FrontMatter -Path $manifestPath } else { '' }
$revalidationSpecRelative = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003.md'
$revalidationDecisionRelative = 'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003-decision.json'
$revalidationSpecPath = Join-Path $projectRoot $revalidationSpecRelative
$revalidationDecisionPath = Join-Path $projectRoot $revalidationDecisionRelative
$manifestStatus = Get-ScalarField -FrontMatter $manifestFront -Name 'status'
$manifestRevalidationId = Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_id'
$manifestRevalidationStatus = Get-ScalarField -FrontMatter $manifestFront -Name 'revalidation_status'
$pendingPhase = $manifestStatus -eq 'pending_revalidation' -and $manifestRevalidationId -eq 'ANG-CR0-REVALIDATION-20260825-003' -and $manifestRevalidationStatus -eq 'PENDING'
$authorizedPhase = $manifestStatus -eq 'authorized' -and $manifestRevalidationId -eq 'ANG-CR0-REVALIDATION-20260825-003' -and $manifestRevalidationStatus -eq 'PASS'

if (-not $pendingPhase -and -not $authorizedPhase) {
    Add-ReleaseError "Manifest is neither the exact PENDING candidate nor the exact authorized/PASS revalidation-003 state: status='$manifestStatus', revalidation_id='$manifestRevalidationId', revalidation_status='$manifestRevalidationStatus'"
}
if ($pendingPhase -and (Test-Path -LiteralPath $revalidationDecisionPath)) {
    Add-ReleaseError 'PENDING revalidation must not have an independent decision artifact'
}
if ($authorizedPhase -and -not (Test-Path -LiteralPath $revalidationDecisionPath -PathType Leaf)) {
    Add-ReleaseError 'Authorized/PASS revalidation requires the independent decision artifact'
}

$manifestExpectations = @{
    release_id = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
    version = '2'
    supersedes_version = '1'
    revalidated_at = '2026-08-25'
    policy = 'ANG-POL-LOCAL-SCAFFOLD-001@1'
    bootstrap_assessment = 'ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1'
    bootstrap_gate = 'ANG-GATE-CONSTRUCTION-RELEASE-0-001@1'
    evidence_scaffold_gate = 'ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001@1'
    evidence_scaffold_disposition = 'SCAFFOLD_ACCEPTED'
    evidence_scaffold_decision_sha256 = '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0'
    normal_evidence_schema_gate_status = 'NOT_RUN'
    normal_resource_design_gate_status = 'NOT_RUN'
    independent_safety_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
    independent_safety_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-001'
    independent_safety_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
    first_leaf_executor = 'ANG-EXEC-CODEX-ROOT-CR0-001'
    resources_leaf_revision = '2'
    resources_leaf_executor = 'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001'
    resources_validator = 'ANG-AUTH-VALIDATOR-001'
    resources_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
    resources_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001'
    resources_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
    resources_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
    revalidation_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
    revalidation_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-003'
    revalidation_reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
    revalidation_decision_path = $revalidationDecisionRelative.Replace('\', '/')
    revalidation_decision_status = $(if ($pendingPhase) { 'ABSENT' } else { 'APPROVED' })
    activation_base_commit = '903f9b9d5e58818d774604dbd6f4d89b2b4544e0'
    formal_human_flourishing_gate_status = 'NOT_RUN'
    slice_status = 'NOT_PASSED'
    milestone_status = 'NOT_PASSED'
}
foreach ($expectation in $manifestExpectations.GetEnumerator()) {
    $actual = Get-ScalarField -FrontMatter $manifestFront -Name $expectation.Key
    if ($actual -ne $expectation.Value) {
        Add-ReleaseError "Manifest $($expectation.Key) mismatch: '$actual' != '$($expectation.Value)'"
    }
}

$archivePath = Join-Path $projectRoot 'work\pre-construction-release-0-20260825.zip'
$archiveHash = '5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3'
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Add-ReleaseError "Rollback archive is missing: $archivePath"
} elseif ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $archiveHash) {
    Add-ReleaseError 'Rollback archive hash mismatch'
}

$assessmentPath = Join-Path $blueprintRoot 'branches\safety\assessments\ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md'
if (Test-Path -LiteralPath $assessmentPath) {
    $assessmentHash = (Get-FileHash -LiteralPath $assessmentPath -Algorithm SHA256).Hash
    $expectedAssessmentHash = Get-ScalarField -FrontMatter $manifestFront -Name 'bootstrap_assessment_sha256'
    if ($assessmentHash -ne $expectedAssessmentHash) {
        Add-ReleaseError "Assessment hash mismatch: $assessmentHash != $expectedAssessmentHash"
    }
}

$pinnedFiles = @{
    'PROJECT_BLUEPRINT.md' = 'A573A281D9733587891AB64B019170B40D41ACC341BA446B11EF764A63A8CE13'
    'docs\blueprints\ROOT_CAPSULE.md' = 'C2EC3B7FBBC04979AF0BA35645F858882021BAB922B26FF21EBFF050DC7C1243'
    'docs\blueprints\HUMAN_FLOURISHING_CONSTITUTION.md' = 'A72B18C5B718829C030C33B7AFCA0F3F53A33232CE17358E6985159B04108EBA'
    'docs\blueprints\BLUEPRINT_INDEX.json' = '1A4EA38741243E24EF46B74DA78EDDE256BDE7E9BE7EDDDAAFA013CF7FF17015'
    'docs\blueprints\PROTOCOL.md' = 'C922170BEDE154056E68732242DD489A738EF64E75453016B816F14D3E02C0CB'
    'docs\blueprints\INTERFACE_REGISTRY.md' = '76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7'
    'docs\blueprints\DEPENDENCY_GRAPH.md' = '6F6079FF224247A1BFE3E4111785C450CD772DF57D11594672379FA78B15CA5D'
    'docs\blueprints\INTEGRATION_SPINE.md' = '78EC4F6909E8D2BDE478FD2557147C0933BE7940F066381995813660A2B53833'
    'docs\blueprints\decisions\ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md' = 'C5B97294FD53AFA9F95E0C28AD6F36C9A7861DF07B50B714F489ED1F37873753'
    'docs\blueprints\decisions\ANG-ADR-0003-CANONICAL-EVIDENCE-IDENTITY.md' = 'CBD8ACDB5EA5D0B217A047DFDC93BF36A36F60A40DE9E2BFE2C97250EF95E20F'
    'docs\blueprints\decisions\ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md' = '90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6'
    'docs\blueprints\releases\construction-0\baselines\ANG-BASELINE-EVIDENCE-SCHEMAS-001.json' = 'F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F'
    'docs\blueprints\branches\safety\BLUEPRINT.md' = '4376A7C61D1CAFFB25671858A79C1A61185721EE850620E4A932B6D5002F8A5D'
    'docs\blueprints\branches\safety\policies\ANG-POL-LOCAL-SCAFFOLD-001.md' = '23D04D544208C7273BA6C7860CC788CDD81640C8DD8236FFD1FED1F2D77495C6'
    'docs\blueprints\branches\safety\assessments\ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md' = '181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC'
    'docs\blueprints\branches\safety\gates\ANG-GATE-CONSTRUCTION-RELEASE-0-001.md' = 'B768F7669241A0C3432E95E0DDB900AE5A007B2369AB3E4D073F473434DE8EEB'
    'docs\blueprints\branches\safety\evidence\ANG-EVID-CR0-SAFETY-DESIGN-001.md' = '0BA8237AB624C980870F263B79EDC1F3974058E8C313F1C422D993FD99FB1F4A'
    'docs\blueprints\branches\evidence\BLUEPRINT.md' = '56DCE997BF6F002BB9202C144913B90D2A28A64203885B8A3730671AFB16ED48'
    'docs\blueprints\branches\evidence\children\evidence-schemas\BLUEPRINT.md' = '2E50B0BB3F016AC0FEA3B5E72FFCDE7E945BD1D82DDA553722986339FF1F93FB'
    'docs\blueprints\branches\evidence\children\evidence-schemas\gates\ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md' = 'A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5'
    'docs\blueprints\branches\evidence\children\evidence-schemas\gates\ANG-GATE-EVIDENCE-SCHEMAS-001.md' = 'CCDB0782B520328AA5B0A04C6684E16EB9390B8338B61FC6BBA1CB8913A49210'
    'docs\blueprints\branches\evidence\children\evidence-schemas\work\ANG-WORK-EVIDENCE-SCHEMAS-001.md' = '5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289'
    'artifacts\control-plane\evidence-schemas\scaffold-gate-decision.json' = '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0'
    'docs\blueprints\branches\resources\BLUEPRINT.md' = '796C1838973BB24B41416968D700DF2FD760A4BA7ECA854E7ECCB4E12B814F53'
    'docs\blueprints\branches\resources\gates\ANG-GATE-CR0-RESOURCES-001.md' = 'AF52436A9E75850201622785206089B81570423B92A73E8C802B283E99F88E0B'
    'docs\blueprints\releases\construction-0\baselines\ANG-BASELINE-CR0-RESOURCES-001.json' = 'EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569'
    'docs\blueprints\branches\worlds\BLUEPRINT.md' = 'B79D0B407F9864AB7C1DB899BA4516507C775179B843B022D91861696A39DAEA'
    'docs\blueprints\branches\science\BLUEPRINT.md' = '854DAB17C6F1CB29DF5466FB18F6D9DDFAAE72039CCC7529A7D86A4F15BF0A40'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-SAFETY-001.md' = 'A7D9C4FC82C9D7C8470B8085BB39F93D64A1B05A9C83614F9B47746860A193EE'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-RESOURCES-001.md' = '4591AA3D673CD9ADCFECFD39CBDE7B8141C127F6F45EFFEBEE09086E35880638'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-WORLDS-001.md' = '3ADCE6430F40D27AA2F5027393898BC448BA3616EE1BAC72E513E1F1AA34E704'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-SCIENCE-001.md' = '363D4848173BE0D6EBAE840BB71D01712D8F4E08891ACAB3DA91F9C841C73798'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-RUNTIME-001.md' = 'A00FF073CB671DA626B991FBC00045AD24FC354E5C21B8B8ABD542367CBC355C'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-LEARNING-001.md' = '5D69A7B0DE3DF06E647ADF538C88E933514ED3F97855D2F10F242BFCEAE83E44'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-TOOLS-001.md' = '092796D9D02DE5E3A561C460257E77E65660D5338EE8E7A26261922BEDB6D551'
    'docs\blueprints\work\slice-00\ANG-WORK-CR0-INTEGRATION-001.md' = '6A8378F9D5241078F80FCEC8895391F1B88731A3C9AFC08AA41E3872DC92B472'
}
foreach ($relativePath in $pinnedFiles.Keys) {
    $absolutePath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        Add-ReleaseError "Pinned input missing: $relativePath"
        continue
    }
    $actualHash = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash
    if ($actualHash -ne $pinnedFiles[$relativePath]) {
        Add-ReleaseError "Pinned input changed: $relativePath"
    }
    if ($manifestRaw -notmatch [regex]::Escape($pinnedFiles[$relativePath])) {
        Add-ReleaseError "Manifest does not record the pinned identity for: $relativePath"
    }
}

foreach ($revalidationPhrase in @(
    'ANG-CR0-REVALIDATION-20260825-001',
    'Revalidation status: **PASS** on 2026-08-25',
    'ANG-CR0-REVALIDATION-20260825-002',
    'is **PASS**',
    "must already be bound in that leaf's frozen front matter and in this manifest",
    'The handoff is an execution output, not a pre-start authority source',
    'record the same frozen executor binding before independent review',
    'Generic `ACCEPTED` is not an alias and is rejected',
    'A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5',
    'No construction leaf or scaffold test has run',
    'two consecutive pending-state rounds produced the same results',
    'Manifest v2 is therefore reauthorized only for its exact frozen scope',
    'host-provided Codex `apply_patch` primitive as the sole authoring mechanism',
    'grants no external-tool status or authority',
    'ANG-ADR-0004',
    'python -B -m unittest tests.unit.evidence.test_evidence_schemas',
    'preserve_on_failure',
    'normal `ANG-GATE-EVIDENCE-SCHEMAS-001`, which remains `NOT_RUN`',
    'ANG-CR0-REVALIDATION-20260825-003',
    'PENDING / NON-AUTHORIZING',
    '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0',
    'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001',
    'ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001',
    'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-003',
    'docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003-decision.json',
    'historical and non-repeatable',
    'no Resources leaf, test, or output has run'
)) {
    if ($manifestRaw -notmatch [regex]::Escape($revalidationPhrase)) {
        Add-ReleaseError "Manifest is missing required revalidation history/current-state text: $revalidationPhrase"
    }
}
if ($manifestRaw -match '(?is)before\s+starting\s+(?:a\s+)?leaf[^.\r\n]*handoff' -or $manifestRaw -match '(?is)before\s+(?:a\s+)?leaf\s+starts[^.\r\n]*(?:bind|authority)[^.\r\n]*handoff') {
    Add-ReleaseError 'Manifest contains a circular pre-start handoff authority/binding requirement'
}
if ($manifestRaw -match '(?is)ANG-CR0-REVALIDATION-20260825-002.{0,240}\*\*PENDING\*\*' -or $manifestRaw -match 'Manifest v2 is not usable authority while revalidation 002 is pending') {
    Add-ReleaseError 'Revalidation 002 remains pending after manifest authorization'
}
if ($pendingPhase -and $manifestRaw -notmatch [regex]::Escape('Manifest v2 is not usable authority while revalidation 003 is PENDING')) {
    Add-ReleaseError 'PENDING manifest lacks the explicit non-authorizing release lock'
}
if ($authorizedPhase -and $manifestRaw -notmatch [regex]::Escape('Revalidation 003 independently APPROVED; Manifest v2 authorized')) {
    Add-ReleaseError 'Authorized manifest lacks the exact post-review authorization statement'
}

$requiredHeadings = @(
    '# Exact objective',
    '## Required context and read set',
    '## Versioned inputs and preconditions',
    '## Exact outputs and authorized write scope',
    '## Non-goals and execution constraints',
    '## Dependencies and status',
    '## Predeclared tests and evidence',
    '## Human-impact mapping and acceptance gate',
    '## Failure, rollback, and handoff'
)
$requiredPhrases = @('outputs/**', 'network', 'package', 'model', 'GPU', 'background', '60 seconds', '10 minutes', '512 MiB', '25 MiB', 'host-provided Codex `apply_patch` primitive', 'Test-Path -LiteralPath', 'Get-FileHash -Algorithm SHA256', 'shell redirection')
$allOutputPaths = [System.Collections.Generic.List[string]]::new()
$seenIds = @{}

foreach ($leafSpec in $leafSpecs) {
    $leafPath = Join-Path $PSScriptRoot $leafSpec.File
    if (-not (Test-Path -LiteralPath $leafPath -PathType Leaf)) {
        Add-ReleaseError "Missing work leaf: $leafPath"
        continue
    }

    $raw = Get-Content -Raw -LiteralPath $leafPath
    $front = Get-FrontMatter -Path $leafPath
    $fieldExpectations = @{
        blueprint_id = $leafSpec.Id
        release_id = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
        revision = $(if ($leafSpec.Id -eq 'ANG-WORK-CR0-RESOURCES-001') { '2' } else { '1' })
        tier = '4'
        design_status = 'approved'
        delivery_status = $leafSpec.Status
        accountable_owner = $leafSpec.Owner
        human_impact_assessment = 'ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1'
        human_flourishing_gate = $(if ($leafSpec.Id -eq 'ANG-WORK-CR0-RESOURCES-001') { 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN' } else { 'ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING' })
    }
    foreach ($expectation in $fieldExpectations.GetEnumerator()) {
        $actual = Get-ScalarField -FrontMatter $front -Name $expectation.Key
        if ($actual -ne $expectation.Value) {
            Add-ReleaseError "$($leafSpec.Id) $($expectation.Key) mismatch: '$actual' != '$($expectation.Value)'"
        }
    }

    $executor = Get-ScalarField -FrontMatter $front -Name 'execution_owner'
    $gate = Get-ScalarField -FrontMatter $front -Name 'gate'
    $rollback = Get-ScalarField -FrontMatter $front -Name 'rollback_ref'
    if ([string]::IsNullOrWhiteSpace($executor) -or $executor -eq 'unassigned') { Add-ReleaseError "$($leafSpec.Id) has no execution owner" }
    if ($leafSpec.Id -eq 'ANG-WORK-CR0-RESOURCES-001' -and $executor -ne 'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001') { Add-ReleaseError "Resources executor is not the exact prebound concrete identity: $executor" }
    if ([string]::IsNullOrWhiteSpace($gate)) { Add-ReleaseError "$($leafSpec.Id) has no gate" }
    if ($leafSpec.Id -eq 'ANG-WORK-CR0-RESOURCES-001') {
        if ($rollback -ne 'ANG-BASELINE-CR0-RESOURCES-001@sha256:EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569') { Add-ReleaseError "$($leafSpec.Id) has the wrong exact Resources rollback reference" }
    } elseif ($rollback -notlike "*$archiveHash*") {
        Add-ReleaseError "$($leafSpec.Id) has the wrong rollback reference"
    }
    if ($seenIds.ContainsKey($leafSpec.Id)) { Add-ReleaseError "Duplicate work-leaf ID: $($leafSpec.Id)" } else { $seenIds[$leafSpec.Id] = $true }
    if ($manifestRaw -notmatch [regex]::Escape($leafSpec.Id)) { Add-ReleaseError "Manifest does not reference $($leafSpec.Id)" }

    foreach ($heading in $requiredHeadings) {
        if ($raw -notmatch "(?m)^$([regex]::Escape($heading))\s*$") {
            Add-ReleaseError "$($leafSpec.Id) missing heading: $heading"
        }
    }
    foreach ($phrase in $requiredPhrases) {
        if ($raw -notmatch [regex]::Escape($phrase)) {
            Add-ReleaseError "$($leafSpec.Id) missing explicit ceiling/prohibition phrase: $phrase"
        }
    }
    if ($raw -notmatch [regex]::Escape("pwsh -NoProfile -NonInteractive -File $($leafSpec.Test)")) {
        Add-ReleaseError "$($leafSpec.Id) missing exact local test command"
    }
    if ($raw -notmatch [regex]::Escape('pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1')) {
        Add-ReleaseError "$($leafSpec.Id) missing blueprint validation command"
    }

    $outputSection = [regex]::Match($raw, '(?ms)^## Exact outputs and authorized write scope\s*\r?\n(?<body>.*?)^## Non-goals and execution constraints')
    if (-not $outputSection.Success) {
        Add-ReleaseError "$($leafSpec.Id) output section is not parseable"
        continue
    }
    $paths = @([regex]::Matches($outputSection.Groups['body'].Value, '(?m)^- `(?<path>[^`]+)`\s*$') | ForEach-Object { $_.Groups['path'].Value })
    if ($paths.Count -lt 2) { Add-ReleaseError "$($leafSpec.Id) has too few literal outputs" }
    foreach ($outputPath in $paths) {
        if ($outputPath -match '[*{}]' -or $outputPath -match '^outputs[\\/]' -or $outputPath -match '^work[\\/]') {
            Add-ReleaseError "$($leafSpec.Id) has non-literal or forbidden output path: $outputPath"
        }
        if ($allOutputPaths.Contains($outputPath)) {
            Add-ReleaseError "Output path assigned to more than one leaf: $outputPath"
        } else {
            $allOutputPaths.Add($outputPath)
        }
    }

    $wordCount = @(($raw -split '\s+') | Where-Object { $_ }).Count
    $approxTokens = [math]::Ceiling($wordCount / 0.75)
    if ($approxTokens -gt 1200) {
        Add-ReleaseError "$($leafSpec.Id) exceeds the 1,200-token work-leaf target (approximately $approxTokens)"
    }
}

$specialLeafChecks = @(
    [pscustomobject]@{
        File = 'ANG-WORK-CR0-WORLDS-001.md'
        Required = @(
            '- `schemas/control/v1/worlds/action.schema.json`',
            '- `tests/synthetic/slice00/worlds/invalid-action.json`',
            '- `tests/synthetic/slice00/worlds/invalid-idempotency.json`',
            '- `tests/synthetic/slice00/worlds/invalid-order.json`',
            '- `tests/synthetic/slice00/worlds/invalid-timeout.json`',
            '- `tests/synthetic/slice00/worlds/invalid-cleanup.json`',
            'conflicting idempotency keys',
            'stale/out-of-order steps',
            'partial-transition cleanup failure',
            'ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001',
            'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json',
            'ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`',
            '90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6'
        )
    },
    [pscustomobject]@{
        File = 'ANG-WORK-CR0-SCIENCE-001.md'
        Required = @(
            'Benchmark-family definitions, sealed partition definitions, and fair-budget definitions are embedded sections of `evaluation-suite.schema.json`',
            'their valid cases live in `valid-control-matrix.json`',
            'their deliberate violations live in the two declared invalid fixtures',
            'No separate benchmark, partition, or budget output is authorized',
            'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json',
            'ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`',
            '90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6'
        )
    },
    [pscustomobject]@{
        File = 'ANG-WORK-CR0-RESOURCES-001.md'
        Required = @(
            'reserved probe-provenance reference fields',
            'RESOURCE-PROBES delivery is deferred to a successor leaf',
            'no probe schema, receipt, fixture, or real hardware probe is authorized',
            'cannot assert that a probe ran',
            'ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001',
            'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json',
            'ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`',
            '90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6'
        )
    },
    [pscustomobject]@{
        File = 'ANG-WORK-CR0-INTEGRATION-001.md'
        Required = @(
            'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json',
            'its exact hash is pinned',
            'ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`',
            '90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6'
        )
    }
)
foreach ($specialLeafCheck in $specialLeafChecks) {
    $specialLeafPath = Join-Path $PSScriptRoot $specialLeafCheck.File
    if (-not (Test-Path -LiteralPath $specialLeafPath -PathType Leaf)) { continue }
    $specialLeafRaw = Get-Content -Raw -LiteralPath $specialLeafPath
    foreach ($requiredText in $specialLeafCheck.Required) {
        if ($specialLeafRaw -notmatch [regex]::Escape($requiredText)) {
            Add-ReleaseError "$($specialLeafCheck.File) is missing a required stabilized-interface constraint: $requiredText"
        }
    }
}

$evidenceDecisionRelative = 'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json'
$evidenceDecisionPath = Join-Path $projectRoot $evidenceDecisionRelative
if (-not (Test-Path -LiteralPath $evidenceDecisionPath -PathType Leaf)) {
    Add-ReleaseError 'Accepted EVIDENCE scaffold decision is missing'
} else {
    $evidenceDecisionHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $evidenceDecisionPath).Hash
    if ($evidenceDecisionHash -ne '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0') {
        Add-ReleaseError "EVIDENCE scaffold decision hash mismatch: $evidenceDecisionHash"
    }
    try {
        $evidenceDecision = Get-Content -Raw -LiteralPath $evidenceDecisionPath | ConvertFrom-Json
        if ($evidenceDecision.disposition -ne 'SCAFFOLD_ACCEPTED') { Add-ReleaseError 'EVIDENCE scaffold decision is not SCAFFOLD_ACCEPTED' }
        if ($evidenceDecision.bindings.release_manifest.sha256 -ne '802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2') { Add-ReleaseError 'EVIDENCE decision lost its historical execution-time manifest binding' }
        if ($evidenceDecision.bindings.leaf.sha256 -ne '5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289') { Add-ReleaseError 'EVIDENCE decision lost its historical leaf binding' }
        if ($evidenceDecision.non_equivalence.normal_evidence_schema_gate.status -ne 'NOT_RUN') { Add-ReleaseError 'EVIDENCE decision improperly changes the normal Evidence gate' }
    } catch {
        Add-ReleaseError "EVIDENCE scaffold decision is not valid JSON: $($_.Exception.Message)"
    }
}

$resourceLeafPath = Join-Path $PSScriptRoot 'ANG-WORK-CR0-RESOURCES-001.md'
$resourceRaw = Get-Content -Raw -LiteralPath $resourceLeafPath
$resourceFront = Get-FrontMatter -Path $resourceLeafPath
$resourceChecks = @{
    revision = '2'
    delivery_status = 'ready'
    activation_state = 'unusable_pending_revalidation'
    execution_owner = 'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001'
    independent_validator = 'ANG-AUTH-VALIDATOR-001'
    independent_gate_authority = 'ANG-AUTH-SAFETY-APPROVER-001'
    independent_gate_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001'
    independent_gate_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
    independent_gate_reviewer_acceptance = 'ACCEPTED'
    independent_gate_reviewer_reachability = 'reachable'
    reviewer_vocabulary_ack = 'ACK_ACCEPTED'
    result_recorder = 'ANG-BP-ROOT'
    authorized_write_scope_owner = 'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001'
    independent_gate_write_scope_owner = 'ANG-AUTH-SAFETY-APPROVER-001'
    independent_gate_write_scope_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001'
    gate = 'ANG-GATE-CR0-RESOURCES-001@1'
    normal_resource_design_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
}
foreach ($check in $resourceChecks.GetEnumerator()) {
    $actual = Get-ScalarField -FrontMatter $resourceFront -Name $check.Key
    if ($actual -ne $check.Value) { Add-ReleaseError "Resources leaf $($check.Key) mismatch: '$actual' != '$($check.Value)'" }
}
if ((Get-ScalarField -FrontMatter $manifestFront -Name 'resources_leaf_executor') -ne (Get-ScalarField -FrontMatter $resourceFront -Name 'execution_owner')) {
    Add-ReleaseError 'Manifest and Resources leaf concrete executor bindings differ'
}

function Get-FrontMatterList {
    param([Parameter(Mandatory)][string]$FrontMatter, [Parameter(Mandatory)][string]$Name)
    $match = [regex]::Match($FrontMatter, "(?ms)^$([regex]::Escape($Name)):\s*\r?\n(?<body>(?:  - [^\r\n]+(?:\r?\n|$))+?)^(?=\S)")
    if (-not $match.Success) { return @() }
    return @([regex]::Matches($match.Groups['body'].Value, '(?m)^  - (?<value>[^\r\n]+)$') | ForEach-Object { $_.Groups['value'].Value.Trim() })
}

$expectedResourceExecutorScope = @(
    'schemas/control/v1/resources/resource-inventory.schema.json',
    'schemas/control/v1/resources/execution-plan.schema.json',
    'tests/synthetic/slice00/resources/constrained.inventory.json',
    'tests/synthetic/slice00/resources/workstation.inventory.json',
    'tests/synthetic/slice00/resources/cluster.inventory.json',
    'tests/synthetic/slice00/resources/invalid-overcommitted.plan.json',
    'tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1',
    'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md'
)
$resourceReceiptPath = 'docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md'
$resourceExecutorScope = @(Get-FrontMatterList -FrontMatter $resourceFront -Name 'authorized_write_scope')
$resourceExecutorDenied = @(Get-FrontMatterList -FrontMatter $resourceFront -Name 'executor_denied_write_scope')
$resourceReviewerScope = @(Get-FrontMatterList -FrontMatter $resourceFront -Name 'independent_gate_write_scope')
if ($resourceExecutorScope.Count -ne 8) { Add-ReleaseError "Resources executor scope must contain exactly eight paths; found $($resourceExecutorScope.Count)" }
foreach ($path in $expectedResourceExecutorScope) { if ($resourceExecutorScope -notcontains $path) { Add-ReleaseError "Resources executor scope omits: $path" } }
if ($resourceExecutorScope -contains $resourceReceiptPath) { Add-ReleaseError 'Resources executor scope contains the independently owned receipt' }
if ($resourceExecutorDenied.Count -ne 1 -or $resourceExecutorDenied[0] -ne $resourceReceiptPath) { Add-ReleaseError 'Resources executor denied scope is not exactly the reviewer receipt' }
if ($resourceReviewerScope.Count -ne 1 -or $resourceReviewerScope[0] -ne $resourceReceiptPath) { Add-ReleaseError 'Resources reviewer write scope is not exactly the reviewer receipt' }
if ([regex]::Matches($resourceRaw, [regex]::Escape('pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1')).Count -ne 2) {
    Add-ReleaseError 'Resources leaf must predeclare exactly two identical Resource-test runs'
}

$resourceGateRelative = 'docs/blueprints/branches/resources/gates/ANG-GATE-CR0-RESOURCES-001.md'
$resourceGatePath = Join-Path $projectRoot $resourceGateRelative
if (-not (Test-Path -LiteralPath $resourceGatePath -PathType Leaf)) {
    Add-ReleaseError 'CR0 Resources gate specification is missing'
} else {
    $resourceGateRaw = Get-Content -Raw -LiteralPath $resourceGatePath
    $resourceGateFront = Get-FrontMatter -Path $resourceGatePath
    $resourceGateChecks = @{
        gate_id = 'ANG-GATE-CR0-RESOURCES-001'
        version = '1'
        status = 'specified'
        activation_state = 'unusable_pending_revalidation'
        leaf = 'ANG-WORK-CR0-RESOURCES-001@2'
        leaf_revision = '2'
        executor = 'ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001'
        independent_validator = 'ANG-AUTH-VALIDATOR-001'
        independent_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001'
        independent_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        reviewer_acceptance = 'ACCEPTED'
        reviewer_reachability = 'reachable'
        reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        allowed_dispositions = 'SCAFFOLD_ACCEPTED|SCAFFOLD_REJECTED|ESCALATE'
        result_recorder = 'ANG-BP-ROOT'
        decision_path = $resourceReceiptPath
        rollback_baseline = 'ANG-BASELINE-CR0-RESOURCES-001@sha256:EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569'
        normal_resource_design_gate = 'ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN'
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
    }
    foreach ($check in $resourceGateChecks.GetEnumerator()) {
        $actual = Get-ScalarField -FrontMatter $resourceGateFront -Name $check.Key
        if ($actual -ne $check.Value) { Add-ReleaseError "Resources gate $($check.Key) mismatch: '$actual' != '$($check.Value)'" }
    }
    foreach ($heading in @('## Claim and non-equivalence','## Entry criteria','## Procedure and precommitted thresholds','## Required negative controls','## Required child gates','## Evidence and decision identity','## Independent authority and write separation','## Failure and rollback','## Waiver policy')) {
        if ($resourceGateRaw -notmatch "(?m)^$([regex]::Escape($heading))\s*$") { Add-ReleaseError "Resources gate missing protocol section: $heading" }
    }
    foreach ($phrase in @('`SCAFFOLD_ACCEPTED` is not a pass','`SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`','Unprefixed generic aliases are invalid','explicitly acknowledged this final vocabulary','writes the handoff first','normal `ANG-GATE-RESOURCE-DESIGN-001`','never broadly or recursively delete')) {
        if ($resourceGateRaw -notmatch [regex]::Escape($phrase)) { Add-ReleaseError "Resources gate missing required role/non-equivalence/rollback phrase: $phrase" }
    }
    $resourceGateResultSemantics = [regex]::Match($resourceGateRaw, '(?ms)^## Claim and non-equivalence\s*\r?\n(?<body>.*?)(?=^## Independent authority and write separation\s*$)').Groups['body'].Value
    if ([string]::IsNullOrWhiteSpace($resourceGateResultSemantics)) {
        Add-ReleaseError 'Resources gate result-semantics region could not be isolated'
    } elseif ($resourceGateResultSemantics -cmatch '(?<!SCAFFOLD_)\b(?:ACCEPTED|REJECTED)\b') {
        Add-ReleaseError 'Resources gate result semantics contain an unprefixed generic ACCEPTED/REJECTED alias'
    }
}

$resourceBaselineRelative = 'docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-001.json'
$resourceBaselinePath = Join-Path $projectRoot $resourceBaselineRelative
$expectedResourceTargets = @($expectedResourceExecutorScope + $resourceReceiptPath)
if (-not (Test-Path -LiteralPath $resourceBaselinePath -PathType Leaf)) {
    Add-ReleaseError 'CR0 Resources absent-state baseline is missing'
} else {
    try {
        $resourceBaseline = Get-Content -Raw -LiteralPath $resourceBaselinePath | ConvertFrom-Json
        if ($resourceBaseline.baseline_id -ne 'ANG-BASELINE-CR0-RESOURCES-001' -or $resourceBaseline.leaf_id -ne 'ANG-WORK-CR0-RESOURCES-001' -or $resourceBaseline.leaf_revision -ne 2) { Add-ReleaseError 'Resources baseline identity/leaf-revision binding is wrong' }
        if ($resourceBaseline.base_commit -ne '903f9b9d5e58818d774604dbd6f4d89b2b4544e0') { Add-ReleaseError 'Resources baseline base commit mismatch' }
        if ($resourceBaseline.pre_release_archive.sha256 -ne $archiveHash) { Add-ReleaseError 'Resources baseline release archive hash mismatch' }
        $baselineTargets = @($resourceBaseline.targets | ForEach-Object { $_.path })
        if ($baselineTargets.Count -ne 9) { Add-ReleaseError "Resources baseline must contain nine targets; found $($baselineTargets.Count)" }
        foreach ($target in @($resourceBaseline.targets)) {
            if ($target.state -ne 'absent') { Add-ReleaseError "Resources baseline target is not absent: $($target.path)" }
            if ($expectedResourceTargets -notcontains $target.path) { Add-ReleaseError "Resources baseline contains undeclared target: $($target.path)" }
        }
        foreach ($path in $expectedResourceTargets) { if ($baselineTargets -notcontains $path) { Add-ReleaseError "Resources baseline omits target: $path" } }
        $restoreOrRemove = @($resourceBaseline.rollback_classes.restore_or_remove)
        $preserveOnFailure = @($resourceBaseline.rollback_classes.preserve_on_failure)
        foreach ($path in $expectedResourceTargets) {
            $classCount = [int]($restoreOrRemove -contains $path) + [int]($preserveOnFailure -contains $path)
            if ($classCount -ne 1) { Add-ReleaseError "Resources baseline target must have exactly one rollback class: $path" }
        }
        foreach ($path in @($resourceReceiptPath,'docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md')) {
            if ($preserveOnFailure -notcontains $path) { Add-ReleaseError "Resources baseline does not preserve failure evidence: $path" }
        }
        if ($resourceBaseline.rollback_rule -notmatch [regex]::Escape('Never broadly or recursively delete')) { Add-ReleaseError 'Resources baseline does not prohibit broad deletion' }
    } catch {
        Add-ReleaseError "Resources baseline is not valid JSON: $($_.Exception.Message)"
    }
}

foreach ($path in $expectedResourceTargets) {
    $absoluteTarget = Join-Path $projectRoot $path
    if (Test-Path -LiteralPath $absoluteTarget) { Add-ReleaseError "Resources prestart target is not absent: $path" }
}
if ($manifestRaw -notmatch '(?m)^\| \[`ANG-WORK-CR0-RESOURCES-001`\].*\| ready \(unusable while revalidation PENDING\) \| yes, after approval \|') {
    Add-ReleaseError 'Manifest does not record Resources as ready but unusable during PENDING revalidation'
}
if ($manifestRaw -notmatch [regex]::Escape('`ANG-WORK-CR0-RESOURCES-001@2`')) {
    Add-ReleaseError 'Manifest does not record the exact Resources leaf revision identity ANG-WORK-CR0-RESOURCES-001@2'
}
if ($manifestRaw -notmatch '(?m)^\| \[`ANG-WORK-EVIDENCE-SCHEMAS-001`\].*\| scaffold_accepted \(historical; non-repeatable\) \|') {
    Add-ReleaseError 'Manifest does not record historical EVIDENCE work as non-repeatable'
}

$validatorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
if ($manifestRaw -notmatch [regex]::Escape($validatorHash)) {
    Add-ReleaseError "Manifest does not pin the current CR0 validator hash: $validatorHash"
}

if (-not (Test-Path -LiteralPath $revalidationSpecPath -PathType Leaf)) {
    Add-ReleaseError 'Revalidation-003 specification is missing'
} else {
    $revalidationSpecRaw = Get-Content -Raw -LiteralPath $revalidationSpecPath
    $revalidationSpecFront = Get-FrontMatter -Path $revalidationSpecPath
    $revalidationSpecChecks = @{
        revalidation_id = 'ANG-CR0-REVALIDATION-20260825-003'
        release_id = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
        manifest_version = '2'
        status = 'PENDING'
        authorization_effect = 'NONE'
        base_commit = '903f9b9d5e58818d774604dbd6f4d89b2b4544e0'
        accountable_owner = 'ANG-BP-ROOT'
        independent_reviewer_role = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-003'
        independent_reviewer_session_ref = 'codex-subagent:/root/flourishing_red_team'
        reviewer_acceptance = 'ACCEPTED'
        reviewer_reachability = 'reachable'
        decision_path = $revalidationDecisionRelative
        decision_status = 'ABSENT'
        pre_activation_manifest_sha256 = '802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2'
        evidence_decision_sha256 = '520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0'
        resources_leaf_revision = '2'
        resources_reviewer_vocabulary_ack = 'ACK_ACCEPTED'
        resources_gate_sha256 = 'AF52436A9E75850201622785206089B81570423B92A73E8C802B283E99F88E0B'
        resources_baseline_sha256 = 'EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569'
        resources_leaf_sha256 = '4591AA3D673CD9ADCFECFD39CBDE7B8141C127F6F45EFFEBEE09086E35880638'
        blueprint_index_sha256 = '1A4EA38741243E24EF46B74DA78EDDE256BDE7E9BE7EDDDAAFA013CF7FF17015'
        root_capsule_sha256 = 'C2EC3B7FBBC04979AF0BA35645F858882021BAB922B26FF21EBFF050DC7C1243'
    }
    foreach ($check in $revalidationSpecChecks.GetEnumerator()) {
        $actual = Get-ScalarField -FrontMatter $revalidationSpecFront -Name $check.Key
        if ($actual -ne $check.Value) { Add-ReleaseError "Revalidation spec $($check.Key) mismatch: '$actual' != '$($check.Value)'" }
    }
    $specValidatorHash = Get-ScalarField -FrontMatter $revalidationSpecFront -Name 'validator_sha256'
    if ($specValidatorHash -ne $validatorHash) { Add-ReleaseError "Revalidation spec validator hash mismatch: '$specValidatorHash' != '$validatorHash'" }
    $candidateManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash
    $specManifestHash = Get-ScalarField -FrontMatter $revalidationSpecFront -Name 'candidate_manifest_sha256'
    if ($pendingPhase -and $specManifestHash -ne $candidateManifestHash) { Add-ReleaseError "Revalidation spec candidate manifest hash mismatch: '$specManifestHash' != '$candidateManifestHash'" }
    foreach ($phrase in @('PENDING / NON-AUTHORIZING','Do not execute the Resources leaf or Resource test','sole future write scope','APPROVED`, `REJECTED`, or `ESCALATE`','SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`','ANG-WORK-CR0-RESOURCES-001@2','`ACK_ACCEPTED`','fulfilled precondition','No circular hash claim')) {
        if ($revalidationSpecRaw -notmatch [regex]::Escape($phrase)) { Add-ReleaseError "Revalidation spec missing required pending/role/circularity phrase: $phrase" }
    }
}

if ($authorizedPhase -and (Test-Path -LiteralPath $revalidationDecisionPath -PathType Leaf)) {
    try {
        $revalidationDecision = Get-Content -Raw -LiteralPath $revalidationDecisionPath | ConvertFrom-Json
        if ($revalidationDecision.revalidation_id -ne 'ANG-CR0-REVALIDATION-20260825-003' -or $revalidationDecision.disposition -ne 'APPROVED') { Add-ReleaseError 'Authorized state lacks exact APPROVED revalidation-003 decision' }
        if ($revalidationDecision.reviewer_role -ne 'ANG-AUTH-SAFETY-APPROVER-001' -or $revalidationDecision.reviewer_instance -ne 'ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-003' -or $revalidationDecision.reviewer_session_ref -ne 'codex-subagent:/root/flourishing_red_team') { Add-ReleaseError 'Revalidation decision reviewer binding mismatch' }
        if ($revalidationDecision.resources_reviewer_vocabulary_ack -ne 'ACK_ACCEPTED') { Add-ReleaseError 'Revalidation decision does not preserve the Resources reviewer ACK_ACCEPTED vocabulary acknowledgement' }
        $specHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $revalidationSpecPath).Hash
        if ($revalidationDecision.bindings.revalidation_spec_sha256 -ne $specHash) { Add-ReleaseError 'Revalidation decision spec hash mismatch' }
        if ($revalidationDecision.bindings.pending_manifest_sha256 -ne (Get-ScalarField -FrontMatter (Get-FrontMatter -Path $revalidationSpecPath) -Name 'candidate_manifest_sha256')) { Add-ReleaseError 'Revalidation decision pending-manifest binding mismatch' }
        if ($revalidationDecision.bindings.resources_leaf_id -ne 'ANG-WORK-CR0-RESOURCES-001@2' -or $revalidationDecision.bindings.resources_leaf_sha256 -ne '4591AA3D673CD9ADCFECFD39CBDE7B8141C127F6F45EFFEBEE09086E35880638' -or $revalidationDecision.bindings.resources_gate_sha256 -ne 'AF52436A9E75850201622785206089B81570423B92A73E8C802B283E99F88E0B' -or $revalidationDecision.bindings.resources_baseline_sha256 -ne 'EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569') { Add-ReleaseError 'Revalidation decision frozen Resources bindings mismatch' }
        $decisionHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $revalidationDecisionPath).Hash
        if ($manifestRaw -notmatch [regex]::Escape($decisionHash)) { Add-ReleaseError 'Authorized manifest does not pin the independent revalidation decision hash' }
    } catch {
        Add-ReleaseError "Independent revalidation decision is not valid JSON: $($_.Exception.Message)"
    }
}

$evidenceLeafPath = Join-Path $blueprintRoot 'branches\evidence\children\evidence-schemas\work\ANG-WORK-EVIDENCE-SCHEMAS-001.md'
if (-not (Test-Path -LiteralPath $evidenceLeafPath -PathType Leaf)) {
    Add-ReleaseError "Missing branch-owned EVIDENCE work leaf: $evidenceLeafPath"
} else {
    $evidenceRaw = Get-Content -Raw -LiteralPath $evidenceLeafPath
    $evidenceFront = Get-FrontMatter -Path $evidenceLeafPath
    $evidenceChecks = @{
        blueprint_id = 'ANG-WORK-EVIDENCE-SCHEMAS-001'
        parent_id = 'ANG-BP-EVIDENCE-SCHEMAS'
        revision = '1'
        tier = '4'
        design_status = 'approved'
        delivery_status = 'ready'
        accountable_owner = 'ANG-AUTH-PROJECT-OWNER-001'
        execution_owner = 'ANG-EXEC-CODEX-ROOT-CR0-001'
        independent_validator = 'ANG-AUTH-VALIDATOR-001'
        independent_gate_authority = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_gate_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-001'
        independent_gate_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        test_id = 'ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001'
        authorized_write_scope_owner = 'ANG-EXEC-CODEX-ROOT-CR0-001'
        independent_gate_write_scope_owner = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_gate_write_scope_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-001'
        gate = 'ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001'
        normal_technical_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
        human_impact_assessment = 'ANG-ASSESS-CONSTRUCTION-RELEASE-0-001'
        authorization_profile = 'ANG-POL-LOCAL-SCAFFOLD-001@1'
        construction_release = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
    }
    foreach ($check in $evidenceChecks.GetEnumerator()) {
        $actual = Get-ScalarField -FrontMatter $evidenceFront -Name $check.Key
        if ($actual -ne $check.Value) {
            Add-ReleaseError "Branch-owned EVIDENCE leaf $($check.Key) mismatch: '$actual' != '$($check.Value)'"
        }
    }
    $manifestFirstExecutor = Get-ScalarField -FrontMatter $manifestFront -Name 'first_leaf_executor'
    $leafExecutionOwner = Get-ScalarField -FrontMatter $evidenceFront -Name 'execution_owner'
    if ([string]::IsNullOrWhiteSpace($manifestFirstExecutor) -or $manifestFirstExecutor -ne $leafExecutionOwner) {
        Add-ReleaseError "Prebound first-leaf executor mismatch: manifest '$manifestFirstExecutor' != leaf '$leafExecutionOwner'"
    }
    if ($null -ne (Get-ScalarField -FrontMatter $evidenceFront -Name 'validation_procedure')) {
        Add-ReleaseError 'Branch-owned EVIDENCE leaf contains the unregistered validation_procedure field'
    }
    if ($manifestRaw -notmatch [regex]::Escape('ANG-WORK-EVIDENCE-SCHEMAS-001')) {
        Add-ReleaseError 'Manifest does not register the branch-owned EVIDENCE leaf'
    }
    $expectedEvidenceRollback = 'ANG-BASELINE-EVIDENCE-SCHEMAS-001@sha256:F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F'
    $actualEvidenceRollbackField = Get-ScalarField -FrontMatter $evidenceFront -Name 'rollback_ref'
    $actualEvidenceRollback = if ($null -eq $actualEvidenceRollbackField) { '' } else { $actualEvidenceRollbackField.Trim('"') }
    if ($actualEvidenceRollback -ne $expectedEvidenceRollback) {
        Add-ReleaseError "Branch-owned EVIDENCE leaf rollback_ref mismatch: '$actualEvidenceRollback'"
    }
    foreach ($phrase in @('60-second command timeout', '600-second aggregate active ceiling', '1 logical CPU core', '512 MiB working set', '25 MiB total new files', 'network, GPU, packages, spend, and background work remain zero')) {
        if ($evidenceRaw -notmatch [regex]::Escape($phrase)) {
            Add-ReleaseError "Branch-owned EVIDENCE leaf missing exact execution ceiling: $phrase"
        }
    }
    foreach ($forbiddenPhrase in @('outputs/**', 'recovered material', 'real-person data', 'no package installation')) {
        if ($evidenceRaw -notmatch [regex]::Escape($forbiddenPhrase)) {
            Add-ReleaseError "Branch-owned EVIDENCE leaf missing explicit scope prohibition: $forbiddenPhrase"
        }
    }
    foreach ($authoringPhrase in @('Codex apply_patch limited to authorized_write_scope', 'Test-Path -LiteralPath', 'Get-FileHash -Algorithm SHA256', 'python -B -m unittest tests.unit.evidence.test_evidence_schemas', '`-B` prevents undeclared bytecode-cache writes', 'Shell redirection, ad-hoc writer scripts, bulk rewrites, cross-role writes, and any other authoring mechanism are unauthorized')) {
        if ($evidenceRaw -notmatch [regex]::Escape($authoringPhrase)) {
            Add-ReleaseError "Branch-owned EVIDENCE leaf missing authoring/command constraint: $authoringPhrase"
        }
    }

    $evidenceWriteScope = [regex]::Match($evidenceFront, '(?ms)^authorized_write_scope:\s*\r?\n(?<body>(?:  - [^\r\n]+(?:\r?\n|$))+?)^(?=\S)')
    $evidenceOutputPaths = @()
    if (-not $evidenceWriteScope.Success) {
        Add-ReleaseError 'Branch-owned EVIDENCE authorized_write_scope is not parseable'
    } else {
        $evidenceOutputPaths = @([regex]::Matches($evidenceWriteScope.Groups['body'].Value, '(?m)^  - (?<path>[^\r\n]+)$') | ForEach-Object { $_.Groups['path'].Value.Trim() })
        if ($evidenceOutputPaths.Count -lt 1) {
            Add-ReleaseError 'Branch-owned EVIDENCE leaf has no literal outputs'
        }
        foreach ($outputPath in $evidenceOutputPaths) {
            if ($outputPath -match '[*{}]' -or $outputPath -match '^outputs[\\/]' -or $outputPath -match '^work[\\/]' -or [System.IO.Path]::IsPathRooted($outputPath)) {
                Add-ReleaseError "Branch-owned EVIDENCE leaf has non-literal or forbidden output path: $outputPath"
            }
            if ($allOutputPaths.Contains($outputPath)) {
                Add-ReleaseError "Output path assigned to more than one leaf: $outputPath"
            } else {
                $allOutputPaths.Add($outputPath)
            }
        }
    }

    $independentDecisionPath = 'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json'
    if ($evidenceOutputPaths -contains $independentDecisionPath) {
        Add-ReleaseError 'Executor-authorized EVIDENCE scope improperly contains the independent gate decision path'
    }
    foreach ($scopePhrase in @(
        "executor_denied_write_scope:`r`n  - $independentDecisionPath",
        'independent_gate_write_scope_owner: ANG-AUTH-SAFETY-APPROVER-001',
        'independent_gate_write_scope_instance: ANG-REVIEW-CODEX-SAFETY-CR0-001',
        "independent_gate_write_scope:`r`n  - $independentDecisionPath"
    )) {
        $normalizedEvidenceRaw = $evidenceRaw -replace "`r`n", "`n"
        $normalizedScopePhrase = $scopePhrase -replace "`r`n", "`n"
        if ($normalizedEvidenceRaw -notmatch [regex]::Escape($normalizedScopePhrase)) {
            Add-ReleaseError "Branch-owned EVIDENCE leaf is missing independent decision scope separation: $normalizedScopePhrase"
        }
    }

    $evidenceBaselinePath = Join-Path $releaseRoot 'baselines\ANG-BASELINE-EVIDENCE-SCHEMAS-001.json'
    if (Test-Path -LiteralPath $evidenceBaselinePath -PathType Leaf) {
        try {
            $evidenceBaseline = Get-Content -Raw -LiteralPath $evidenceBaselinePath | ConvertFrom-Json
            if ($evidenceBaseline.baseline_id -ne 'ANG-BASELINE-EVIDENCE-SCHEMAS-001' -or $evidenceBaseline.leaf_id -ne 'ANG-WORK-EVIDENCE-SCHEMAS-001') {
                Add-ReleaseError 'EVIDENCE baseline identity or leaf binding is wrong'
            }
            if ($evidenceBaseline.pre_release_archive.path -ne 'work/pre-construction-release-0-20260825.zip' -or $evidenceBaseline.pre_release_archive.sha256 -ne $archiveHash) {
                Add-ReleaseError 'EVIDENCE baseline has the wrong release rollback identity'
            }
            $baselineTargets = @($evidenceBaseline.targets | ForEach-Object { $_.path })
            foreach ($target in @($evidenceBaseline.targets)) {
                if ($target.state -ne 'absent') {
                    Add-ReleaseError "EVIDENCE baseline target is not an exact absent-state record: $($target.path)"
                }
            }
            foreach ($outputPath in $evidenceOutputPaths) {
                if ($baselineTargets -notcontains $outputPath) {
                    Add-ReleaseError "EVIDENCE output is missing from the exact baseline: $outputPath"
                }
            }
            foreach ($baselineTarget in $baselineTargets) {
                if ($evidenceOutputPaths -notcontains $baselineTarget -and $baselineTarget -ne $independentDecisionPath) {
                    Add-ReleaseError "EVIDENCE baseline contains an undeclared output: $baselineTarget"
                }
            }
            if ($baselineTargets -notcontains $independentDecisionPath) {
                Add-ReleaseError 'EVIDENCE baseline omits the independently owned gate decision path'
            }

            $restoreOrRemove = @($evidenceBaseline.rollback_classes.restore_or_remove)
            $preserveOnFailure = @($evidenceBaseline.rollback_classes.preserve_on_failure)
            $expectedPreserved = @(
                'artifacts/control-plane/evidence-schemas/test-receipt.json',
                'artifacts/control-plane/evidence-schemas/effect-receipt.json',
                'artifacts/control-plane/evidence-schemas/HANDOFF.md',
                $independentDecisionPath
            )
            foreach ($preservedPath in $expectedPreserved) {
                if ($preserveOnFailure -notcontains $preservedPath) {
                    Add-ReleaseError "EVIDENCE baseline does not classify immutable evidence preserve_on_failure: $preservedPath"
                }
                if ($restoreOrRemove -contains $preservedPath) {
                    Add-ReleaseError "EVIDENCE baseline gives immutable evidence a destructive rollback class: $preservedPath"
                }
            }
            foreach ($baselineTarget in $baselineTargets) {
                $classCount = 0
                if ($restoreOrRemove -contains $baselineTarget) { $classCount++ }
                if ($preserveOnFailure -contains $baselineTarget) { $classCount++ }
                if ($classCount -ne 1) {
                    Add-ReleaseError "EVIDENCE baseline target must have exactly one rollback class: $baselineTarget"
                }
            }
            if ($evidenceBaseline.rollback_rule -notmatch [regex]::Escape('Never broadly or recursively delete') -or $evidenceBaseline.rollback_rule -notmatch [regex]::Escape('never delete or rewrite preserve_on_failure evidence')) {
                Add-ReleaseError 'EVIDENCE baseline does not preserve failure evidence or prohibit broad deletion'
            }

            if ($allOutputPaths.Contains($independentDecisionPath)) {
                Add-ReleaseError "Output path assigned to more than one role/leaf: $independentDecisionPath"
            } else {
                $allOutputPaths.Add($independentDecisionPath)
            }
        } catch {
            Add-ReleaseError "EVIDENCE baseline is not valid JSON: $($_.Exception.Message)"
        }
    }
    if ($manifestRaw -notmatch '(?m)^\| \[`ANG-WORK-EVIDENCE-SCHEMAS-001`\].*\| scaffold_accepted \(historical; non-repeatable\) \| no; already executed \|') {
        Add-ReleaseError 'Manifest does not mark the branch-owned EVIDENCE execution historical and non-repeatable'
    }
    if ($manifestRaw -notmatch '(?m)^\| \[`ANG-WORK-EVIDENCE-SCHEMAS-001`\].*\| `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` \|') {
        Add-ReleaseError 'Manifest does not route EVIDENCE bootstrap progression through the CR0 scaffold gate'
    }
}

$scaffoldGatePath = Join-Path $blueprintRoot 'branches\evidence\children\evidence-schemas\gates\ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md'
$normalEvidenceGatePath = Join-Path $blueprintRoot 'branches\evidence\children\evidence-schemas\gates\ANG-GATE-EVIDENCE-SCHEMAS-001.md'
if (Test-Path -LiteralPath $scaffoldGatePath -PathType Leaf) {
    $scaffoldGateRaw = Get-Content -Raw -LiteralPath $scaffoldGatePath
    $scaffoldGateFront = Get-FrontMatter -Path $scaffoldGatePath
    $scaffoldGateChecks = @{
        gate_id = 'ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001'
        version = '1'
        status = 'specified'
        gate_class = 'bootstrap_scaffold_acceptance'
        release = 'ANG-CR-0001-CONSTRUCTION-RELEASE-0'
        independent_verifier = 'ANG-AUTH-VALIDATOR-001'
        independent_safety_reviewer = 'ANG-AUTH-SAFETY-APPROVER-001'
        independent_reviewer_instance = 'ANG-REVIEW-CODEX-SAFETY-CR0-001'
        independent_reviewer_session_ref = 'codex-subagent:/root/safety_change_map'
        test_id = 'ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001'
        result_recorder = 'ANG-BP-ROOT'
        decision_writer = 'ANG-AUTH-SAFETY-APPROVER-001'
        decision_path = 'artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json'
        executor = 'ANG-EXEC-CODEX-ROOT-CR0-001'
        human_impact_assessment = 'ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1'
        human_flourishing_gate = 'ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN'
        normal_technical_gate = 'ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN'
    }
    foreach ($check in $scaffoldGateChecks.GetEnumerator()) {
        $actual = Get-ScalarField -FrontMatter $scaffoldGateFront -Name $check.Key
        if ($actual -ne $check.Value) {
            Add-ReleaseError "EVIDENCE scaffold gate $($check.Key) mismatch: '$actual' != '$($check.Value)'"
        }
    }
    foreach ($phrase in @(
        'A `SCAFFOLD_ACCEPTED` disposition does **not** pass `ANG-GATE-EVIDENCE-SCHEMAS-001`',
        'must not contain a gate disposition',
        'Only bound reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-001`',
        '`reviewer_role`, `reviewer_instance`, `reviewer_session_ref`',
        "The executor's write scope excludes this file",
        'The gate specification itself is never edited to record a run result',
        'exactly `SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`'
    )) {
        if ($scaffoldGateRaw -notmatch [regex]::Escape($phrase)) {
            Add-ReleaseError "EVIDENCE scaffold gate is missing role/disposition non-equivalence text: $phrase"
        }
    }
    if ($scaffoldGateRaw -cmatch '(?<!SCAFFOLD_)\bACCEPTED\b') {
        Add-ReleaseError 'EVIDENCE scaffold gate uses forbidden generic ACCEPTED disposition alias'
    }
} else {
    Add-ReleaseError "Missing EVIDENCE scaffold gate: $scaffoldGatePath"
}

if (Test-Path -LiteralPath $normalEvidenceGatePath -PathType Leaf) {
    $normalGateRaw = Get-Content -Raw -LiteralPath $normalEvidenceGatePath
    $normalGateFront = Get-FrontMatter -Path $normalEvidenceGatePath
    if ((Get-ScalarField -FrontMatter $normalGateFront -Name 'gate_id') -ne 'ANG-GATE-EVIDENCE-SCHEMAS-001' -or (Get-ScalarField -FrontMatter $normalGateFront -Name 'status') -ne 'specified_not_run') {
        Add-ReleaseError 'Normal EVIDENCE schema gate is not explicitly specified_not_run'
    }
    foreach ($phrase in @('normal technical completion gate', 'remains `specified_not_run` during CR0 bootstrap construction', 'cannot inherit or be satisfied by `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`')) {
        if ($normalGateRaw -notmatch [regex]::Escape($phrase)) {
            Add-ReleaseError "Normal EVIDENCE gate is missing bootstrap non-equivalence text: $phrase"
        }
    }
} else {
    Add-ReleaseError "Missing normal EVIDENCE schema gate: $normalEvidenceGatePath"
}

foreach ($deferredId in @('ANG-WORK-CR0-RUNTIME-001', 'ANG-WORK-CR0-LEARNING-001', 'ANG-WORK-CR0-TOOLS-001')) {
    $escapedId = [regex]::Escape($deferredId)
    if ($manifestRaw -notmatch "(?s)$escapedId.*?not_ready.*?\| no \|") {
        Add-ReleaseError "$deferredId is not explicitly excluded from CR0 activation"
    }
}

$actualLeafFiles = @(Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter 'ANG-WORK-CR0-*.md')
if ($actualLeafFiles.Count -ne $leafSpecs.Count) {
    Add-ReleaseError "Expected $($leafSpecs.Count) work leaves; found $($actualLeafFiles.Count)"
}
if ($allOutputPaths.Count -ne 100) {
    Add-ReleaseError "Expected exactly 100 unique role-owned CR0 outputs; found $($allOutputPaths.Count)"
}

if ($errors.Count -gt 0) {
    Write-Host "Construction Release 0 validation FAILED with $($errors.Count) error(s):"
    foreach ($validationError in $errors) { Write-Host "- $validationError" }
    exit 1
}

if ($pendingPhase) {
    Write-Host "Construction Release 0 structural validation PASS - PENDING / NON-AUTHORIZING: $($leafSpecs.Count) release-scoped leaves plus one historical branch-owned EVIDENCE leaf, $($allOutputPaths.Count) unique declared outputs, exact Resources absence/roles/rollback and semantic inputs verified. This result grants no leaf execution permission."
} else {
    Write-Host "Construction Release 0 validation PASS - revalidation 003 independently APPROVED and manifest authorized for the exact frozen scope: $($leafSpecs.Count) release-scoped leaves, one historical EVIDENCE leaf, and $($allOutputPaths.Count) unique declared outputs."
}
exit 0
