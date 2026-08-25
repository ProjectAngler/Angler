[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
$schemaRoot = Join-Path $projectRoot 'schemas\control\v1\resources'
$fixtureRoot = $PSScriptRoot
$failures = [System.Collections.Generic.List[string]]::new()
$caseCount = 0

function Read-Json {
    param([Parameter(Mandatory)][string]$Path)
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json -Depth 100
}

function Copy-JsonObject {
    param([Parameter(Mandatory)]$Value)
    return $Value | ConvertTo-Json -Depth 100 -Compress | ConvertFrom-Json -Depth 100
}

function Add-CaseResult {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]$Actual,
        [Parameter(Mandatory)][bool]$Expected
    )
    $script:caseCount++
    if ($Actual -ne $Expected) {
        $script:failures.Add("$Name expected=$Expected actual=$Actual")
    }
}

function Test-DigestRef {
    param($Value)
    return $Value -is [string] -and $Value -match '^sha256:[0-9a-f]{64}$'
}

function Test-Inventory {
    param($Inventory)
    try {
        if ($Inventory.contract.id -ne 'ANG-CTR-RESOURCE-INVENTORY-001' -or $Inventory.contract.version -ne '1.0.0') { return $false }
        if (-not (Test-DigestRef $Inventory.inventory_id) -or -not (Test-DigestRef $Inventory.evidence_envelope_ref)) { return $false }
        if ($Inventory.synthetic -ne $true -or $Inventory.measured -ne $false -or $Inventory.measurement_status -ne 'SYNTHETIC_NOT_MEASURED') { return $false }
        if (@('CONSTRAINED', 'WORKSTATION', 'CLUSTER') -notcontains $Inventory.profile_tier) { return $false }
        if ($Inventory.topology_version -lt 1 -or $Inventory.capacity.cpu_logical_cores -lt 1 -or $Inventory.capacity.ram_bytes -lt 1073741824) { return $false }
        if ($Inventory.administrative_ceiling.cpu_cores -gt $Inventory.capacity.cpu_logical_cores) { return $false }
        if ($Inventory.administrative_ceiling.ram_bytes -gt $Inventory.capacity.ram_bytes) { return $false }
        if ($Inventory.administrative_ceiling.storage_bytes -gt $Inventory.capacity.storage_working_bytes) { return $false }
        if ($Inventory.administrative_ceiling.accelerator_count -gt @($Inventory.capacity.accelerators).Count) { return $false }
        $physicalAcceleratorMemory = 0
        foreach ($accelerator in @($Inventory.capacity.accelerators)) {
            if ($accelerator.kind -ne 'SYNTHETIC_ACCELERATOR' -or $accelerator.memory_bytes -lt 1073741824) { return $false }
            $physicalAcceleratorMemory += $accelerator.memory_bytes
        }
        if ($Inventory.administrative_ceiling.accelerator_memory_bytes -gt $physicalAcceleratorMemory) { return $false }
        foreach ($probeRef in @($Inventory.probe_provenance_refs)) {
            if ($probeRef -notmatch '^ang-evid:probe:[A-Za-z0-9._-]+$') { return $false }
        }
        return $true
    } catch {
        return $false
    }
}

function New-ValidPlan {
    param($Inventory)
    $hasAccelerator = @($Inventory.capacity.accelerators).Count -gt 0
    return [pscustomobject]@{
        contract = [pscustomobject]@{ id = 'ANG-CTR-EXECUTION-PLAN-001'; version = '1.0.0' }
        plan_id = 'sha256:5555555555555555555555555555555555555555555555555555555555555555'
        evidence_envelope_ref = 'sha256:a555555555555555555555555555555555555555555555555555555555555555'
        inventory_ref = $Inventory.inventory_id
        evidence_ref = 'sha256:b555555555555555555555555555555555555555555555555555555555555555'
        synthetic = $true
        plan_state = 'PLANNED_NOT_EXECUTED'
        measured = $false
        workload = [pscustomobject]@{ model_profile_ref = 'synthetic-profile:bounded'; precision = 'INT4'; plastic_state_topology_ref = 'synthetic-topology:single-state' }
        topology_binding = [pscustomobject]@{ inventory_topology_version = $Inventory.topology_version; change_policy = 'ABORT_AT_TRANSACTION_BOUNDARY' }
        resource_request = [pscustomobject]@{ cpu_cores = 1; ram_bytes = 1073741824; accelerator_count = [int]$hasAccelerator; accelerator_memory_bytes = $(if ($hasAccelerator) { 1073741824 } else { 0 }); storage_bytes = 1073741824 }
        reservations = [pscustomobject]@{
            host_headroom = [pscustomobject]@{ ram_bytes = 536870912; storage_bytes = 536870912 }
            evaluation_headroom = [pscustomobject]@{ ram_bytes = 536870912; storage_bytes = 536870912 }
            rollback_headroom = [pscustomobject]@{ ram_bytes = 536870912; storage_bytes = 536870912 }
        }
        placement = [pscustomobject]@{ mode = $(if ($hasAccelerator) { 'SINGLE_ACCELERATOR' } else { 'CPU_ONLY' }); parallelism = 1; offload = $false }
        objective = [pscustomobject]@{ primary = 'QUALITY'; tie_breaker = 'TIME'; externally_selected = $true }
        probe_provenance_refs = @()
        routing_policy = [pscustomobject]@{ query_conditioned_model_or_adapter_selection = $false }
        replanning_policy = [pscustomobject]@{ mid_transaction_change = $false }
        fallback = [pscustomobject]@{ on_infeasible = 'NO_PLAN'; on_drift = 'ABORT_AND_PRESERVE_STATE' }
        extensions = [pscustomobject]@{}
    }
}

function Test-Plan {
    param($Plan, $Inventory)
    try {
        if ($Plan.contract.id -ne 'ANG-CTR-EXECUTION-PLAN-001' -or $Plan.contract.version -ne '1.0.0') { return $false }
        if (-not (Test-DigestRef $Plan.plan_id) -or -not (Test-DigestRef $Plan.inventory_ref) -or -not (Test-DigestRef $Plan.evidence_ref)) { return $false }
        if ($Plan.inventory_ref -ne $Inventory.inventory_id -or $Plan.synthetic -ne $true -or $Plan.measured -ne $false -or $Plan.plan_state -ne 'PLANNED_NOT_EXECUTED') { return $false }
        if ($Plan.inventory_ref -is [array] -or $Plan.evidence_ref -is [array]) { return $false }
        if ($Plan.routing_policy.query_conditioned_model_or_adapter_selection -ne $false -or $Plan.replanning_policy.mid_transaction_change -ne $false) { return $false }
        if ($Plan.topology_binding.inventory_topology_version -ne $Inventory.topology_version -or $Plan.topology_binding.change_policy -ne 'ABORT_AT_TRANSACTION_BOUNDARY') { return $false }
        foreach ($reservationName in @('host_headroom', 'evaluation_headroom', 'rollback_headroom')) {
            $reservation = $Plan.reservations.$reservationName
            if ($reservation.ram_bytes -le 0 -or $reservation.storage_bytes -le 0) { return $false }
        }
        $reservedRam = $Plan.reservations.host_headroom.ram_bytes + $Plan.reservations.evaluation_headroom.ram_bytes + $Plan.reservations.rollback_headroom.ram_bytes
        $reservedStorage = $Plan.reservations.host_headroom.storage_bytes + $Plan.reservations.evaluation_headroom.storage_bytes + $Plan.reservations.rollback_headroom.storage_bytes
        if ($Plan.resource_request.cpu_cores -gt $Inventory.administrative_ceiling.cpu_cores) { return $false }
        if (($Plan.resource_request.ram_bytes + $reservedRam) -gt $Inventory.administrative_ceiling.ram_bytes) { return $false }
        if (($Plan.resource_request.storage_bytes + $reservedStorage) -gt $Inventory.administrative_ceiling.storage_bytes) { return $false }
        if ($Plan.resource_request.accelerator_count -gt $Inventory.administrative_ceiling.accelerator_count) { return $false }
        if ($Plan.resource_request.accelerator_memory_bytes -gt $Inventory.administrative_ceiling.accelerator_memory_bytes) { return $false }
        foreach ($probeRef in @($Plan.probe_provenance_refs)) {
            if ($probeRef -notmatch '^ang-evid:probe:[A-Za-z0-9._-]+$') { return $false }
        }
        if ($Plan.PSObject.Properties.Name -contains 'probe_success') { return $false }
        return $true
    } catch {
        return $false
    }
}

$inventorySchema = Read-Json (Join-Path $schemaRoot 'resource-inventory.schema.json')
$planSchema = Read-Json (Join-Path $schemaRoot 'execution-plan.schema.json')
$constrained = Read-Json (Join-Path $fixtureRoot 'constrained.inventory.json')
$workstation = Read-Json (Join-Path $fixtureRoot 'workstation.inventory.json')
$cluster = Read-Json (Join-Path $fixtureRoot 'cluster.inventory.json')
$overcommitted = Read-Json (Join-Path $fixtureRoot 'invalid-overcommitted.plan.json')

Add-CaseResult 'inventory schema identity and synthetic guard' ($inventorySchema.title -eq 'ANG-CTR-RESOURCE-INVENTORY-001@1.0.0' -and $inventorySchema.properties.synthetic.const -eq $true -and $inventorySchema.properties.measured.const -eq $false) $true
Add-CaseResult 'execution plan schema identity and routing guard' ($planSchema.title -eq 'ANG-CTR-EXECUTION-PLAN-001@1.0.0' -and $planSchema.properties.routing_policy.properties.query_conditioned_model_or_adapter_selection.const -eq $false) $true
Add-CaseResult 'constrained inventory accepted' (Test-Inventory $constrained) $true
Add-CaseResult 'workstation inventory accepted' (Test-Inventory $workstation) $true
Add-CaseResult 'cluster inventory accepted' (Test-Inventory $cluster) $true

$validPlan = New-ValidPlan $constrained
Add-CaseResult 'bounded plan accepted' (Test-Plan $validPlan $constrained) $true
Add-CaseResult 'overcommitted fixture rejected' (Test-Plan $overcommitted $constrained) $false

$missingHeadroom = Copy-JsonObject $validPlan
$missingHeadroom.reservations.rollback_headroom.ram_bytes = 0
Add-CaseResult 'missing rollback headroom rejected' (Test-Plan $missingHeadroom $constrained) $false

$missingInventory = Copy-JsonObject $validPlan
$missingInventory.inventory_ref = $null
Add-CaseResult 'missing inventory identity rejected' (Test-Plan $missingInventory $constrained) $false

$multipleEvidence = Copy-JsonObject $validPlan
$multipleEvidence.evidence_ref = @($validPlan.evidence_ref, 'sha256:b666666666666666666666666666666666666666666666666666666666666666')
Add-CaseResult 'multiple evidence identities rejected' (Test-Plan $multipleEvidence $constrained) $false

$fabricatedMeasurement = Copy-JsonObject $validPlan
$fabricatedMeasurement.measured = $true
Add-CaseResult 'fabricated measured plan rejected' (Test-Plan $fabricatedMeasurement $constrained) $false

$fabricatedProbe = Copy-JsonObject $validPlan
$fabricatedProbe | Add-Member -NotePropertyName probe_success -NotePropertyValue $true
Add-CaseResult 'fabricated probe success rejected' (Test-Plan $fabricatedProbe $constrained) $false

$queryRouting = Copy-JsonObject $validPlan
$queryRouting.routing_policy.query_conditioned_model_or_adapter_selection = $true
Add-CaseResult 'query conditioned routing rejected' (Test-Plan $queryRouting $constrained) $false

$midTransaction = Copy-JsonObject $validPlan
$midTransaction.replanning_policy.mid_transaction_change = $true
Add-CaseResult 'mid transaction topology change rejected' (Test-Plan $midTransaction $constrained) $false

$malformedProbe = Copy-JsonObject $validPlan
$malformedProbe.probe_provenance_refs = @('probe-success:true')
Add-CaseResult 'malformed probe provenance rejected' (Test-Plan $malformedProbe $constrained) $false

$nonSynthetic = Copy-JsonObject $workstation
$nonSynthetic.synthetic = $false
Add-CaseResult 'non-synthetic inventory rejected' (Test-Inventory $nonSynthetic) $false

if ($failures.Count -gt 0) {
    Write-Output "FAIL - CR0 Resources scaffold: $caseCount cases, $($failures.Count) failure(s)."
    foreach ($failure in $failures) { Write-Output "- $failure" }
    exit 1
}

Write-Output "PASS - CR0 Resources scaffold: $caseCount cases, 0 failures; 3 synthetic tiers accepted and all declared negative controls rejected."
exit 0
