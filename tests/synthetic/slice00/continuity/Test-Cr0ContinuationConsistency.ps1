$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$failures = [System.Collections.Generic.List[string]]::new()
$cases = 0

function Check([bool]$condition, [string]$name) {
    $script:cases++
    if (-not $condition) { $script:failures.Add($name) }
}
function Text([string]$relative) { [IO.File]::ReadAllText((Join-Path $repo $relative)) }
function Hash([string]$relative) { (Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash }

$immutable = [ordered]@{
    'src/angler/episodes/__init__.py'='62E298AD19F52B6A620C4D62B67116B2A86B6294C68B7CD2A0B9C60BC1D6A0FE'
    'src/angler/episodes/canonical.py'='E0E1B85D4C00CED3BB90917A4C37D3D3950ACAC9A1DB7FAA47A6AA897E869A72'
    'src/angler/episodes/schema_validation.py'='74F9B5F3FFE248D8E0B667AF1774BE92E29F55F89E87C2983C4431017611AF3D'
    'src/angler/episodes/visibility.py'='DBD94B64D196665048479AF4EA1B902525B2E591601B01BEB51D5C00D7F290D3'
    'src/angler/episodes/schemas/evidence-envelope.v1.json'='E8762C3576D4DFCAF46B833905A589551089A5B606054D273EFEC31B887D6CB6'
    'src/angler/episodes/schemas/episode.v1.json'='B9925BB0B37B43AC508535DA9635A40766C63FBBD36BEA2D04168EEAD7D79DBE'
    'src/angler/episodes/schemas/experiment-manifest.v1.json'='4002E8FA87BEE91365A6E3CA926EEC758F7F53992DBAB2D7546A6CCA3CEFA9DF'
    'tests/unit/evidence/test_evidence_schemas.py'='12C5014F6168D36A3D16B12B1103AACB5A2DE8DD3056D5C2325C5229844CAD5D'
}
foreach ($entry in $immutable.GetEnumerator()) { Check ((Hash $entry.Key) -eq $entry.Value) "immutable:$($entry.Key)" }

$projections = @('docs/blueprints/ROOT_CAPSULE.md','docs/blueprints/STATUS.md','docs/blueprints/branches/resources/CAPSULE.md','docs/blueprints/branches/resources/STATUS.md','docs/blueprints/branches/evidence/BLUEPRINT.md','docs/blueprints/branches/evidence/CAPSULE.md','docs/blueprints/branches/evidence/STATUS.md','docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md','docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md','docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md','docs/blueprints/BLUEPRINT_INDEX.json','docs/blueprints/TREE.md','docs/blueprints/TRACEABILITY.md')
foreach ($path in $projections) { Check (Test-Path -LiteralPath (Join-Path $repo $path) -PathType Leaf) "projection:$path" }

$all = ($projections | ForEach-Object { Text $_ }) -join "`n"
Check ($all.Contains('520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0')) 'evidence-decision-projected'
Check ($all.Contains('D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B')) 'resources-receipt-projected'
Check ($all.Contains('ANG-WORK-CR0-CONTINUITY-002@1')) 'continuity-leaf-projected'
Check ($all.Contains('NOT_RUN')) 'normal-gates-not-run'
Check ($all.Contains('NOT_PASSED')) 'milestones-not-passed'
Check (-not $all.Contains('revalidation 004 is PENDING')) 'no-stale-revalidation-pending'
Check (-not $all.Contains('first schema leaf ready')) 'no-stale-schema-ready'

$index = Text 'docs/blueprints/BLUEPRINT_INDEX.json' | ConvertFrom-Json
$evidence = $index.nodes | Where-Object id -eq 'ANG-BP-EVIDENCE'
$schemas = $index.nodes | Where-Object id -eq 'ANG-BP-EVIDENCE-SCHEMAS'
$resources = $index.nodes | Where-Object id -eq 'ANG-BP-RESOURCES'
Check ($evidence.revision -eq 3) 'index-evidence-revision'
Check ($schemas.revision -eq 2) 'index-schemas-revision'
Check ($resources.delivery_status -eq 'cr0_scaffold_accepted_normal_gate_not_run') 'index-resources-delivery'

$addendum = Text 'docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md'
Check ($addendum.Contains('903f9b9d5e58818d774604dbd6f4d89b2b4544e0')) 'addendum-accepted-commit'
Check ($addendum.Contains('must not be rerun')) 'addendum-non-repeatable'

if ($failures.Count -gt 0) {
    Write-Output "FAIL cases=$cases failures=$($failures.Count)"
    $failures | ForEach-Object { Write-Output "FAIL $_" }
    exit 1
}
Write-Output "PASS cases=$cases failures=0"
