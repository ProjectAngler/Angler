[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$blueprintRoot = Join-Path $projectRoot 'docs\blueprints'
$indexPath = Join-Path $blueprintRoot 'BLUEPRINT_INDEX.json'
$errors = [System.Collections.Generic.List[string]]::new()

function Add-BlueprintError {
    param([Parameter(Mandatory)][string]$Message)
    $errors.Add($Message)
}

function Get-FrontMatter {
    param([Parameter(Mandatory)][string]$Path)
    $raw = Get-Content -Raw -LiteralPath $Path
    $match = [regex]::Match($raw, '\A---\s*\r?\n(?<front>.*?)\r?\n---\s*\r?\n', 'Singleline')
    if (-not $match.Success) {
        Add-BlueprintError "Missing front matter: $Path"
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

function Get-ListField {
    param(
        [Parameter(Mandatory)][string]$FrontMatter,
        [Parameter(Mandatory)][string]$Name
    )
    $match = [regex]::Match(
        $FrontMatter,
        "(?ms)^$([regex]::Escape($Name)):\s*\r?\n(?<body>(?:  - [^\r\n]+\r?\n?)*)"
    )
    if (-not $match.Success) { return @() }
    return @([regex]::Matches($match.Groups['body'].Value, '(?m)^  - (?<value>[^\r\n]+)$') |
        ForEach-Object { $_.Groups['value'].Value.Trim() })
}

if (-not (Test-Path -LiteralPath $indexPath)) {
    throw "Blueprint index not found: $indexPath"
}

$index = Get-Content -Raw -LiteralPath $indexPath | ConvertFrom-Json
$nodeById = @{}
foreach ($concreteNode in $index.nodes) { $nodeById[$concreteNode.id] = $concreteNode }
$concreteIds = @($index.nodes | ForEach-Object { $_.id })
$declaredIds = @($index.nodes |
    Where-Object { $_.PSObject.Properties.Name -contains 'declared_children' } |
    ForEach-Object { $_.declared_children } |
    ForEach-Object { $_.id })
$allIds = @($concreteIds + $declaredIds)

foreach ($duplicate in @($allIds | Group-Object | Where-Object Count -gt 1)) {
    Add-BlueprintError "Duplicate blueprint ID: $($duplicate.Name)"
}

$roots = @($index.nodes | Where-Object { $null -eq $_.parent_id })
if ($roots.Count -ne 1) {
    Add-BlueprintError "Expected exactly one concrete root; found $($roots.Count)"
}

foreach ($node in $index.nodes) {
    if ($null -ne $node.parent_id -and $concreteIds -notcontains $node.parent_id) {
        Add-BlueprintError "Unknown parent $($node.parent_id) for $($node.id)"
    }

    $concreteChildIds = @()
    if ($node.PSObject.Properties.Name -contains 'children') {
        $concreteChildIds = @($node.children)
    }
    foreach ($childId in $concreteChildIds) {
        if ($concreteIds -notcontains $childId) {
            Add-BlueprintError "Unknown concrete child $childId declared by $($node.id)"
            continue
        }
        if ($nodeById[$childId].parent_id -ne $node.id) {
            Add-BlueprintError "Concrete child parent mismatch for ${childId}: $($nodeById[$childId].parent_id) != $($node.id)"
        }
    }

    $stubChildIds = @()
    if ($node.PSObject.Properties.Name -contains 'declared_children') {
        $stubChildIds = @($node.declared_children | ForEach-Object { $_.id })
    }
    foreach ($childId in $stubChildIds) {
        if ($concreteIds -contains $childId) {
            Add-BlueprintError "Child $childId is both concrete and declared as a stub by $($node.id)"
        }
    }

    foreach ($field in @('blueprint_path', 'capsule_path', 'status_path')) {
        if (-not ($node.PSObject.Properties.Name -contains $field)) { continue }
        $relativePath = $node.$field
        if ([string]::IsNullOrWhiteSpace($relativePath)) { continue }
        $candidate = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $relativePath))
        if (-not (Test-Path -LiteralPath $candidate)) {
            Add-BlueprintError "Missing $field for $($node.id): $relativePath"
        }
    }

    foreach ($field in @('contract_paths', 'gate_paths')) {
        if (-not ($node.PSObject.Properties.Name -contains $field)) { continue }
        foreach ($relativePath in $node.$field) {
            $candidate = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $relativePath))
            if (-not (Test-Path -LiteralPath $candidate)) {
                Add-BlueprintError "Missing $field artifact for $($node.id): $relativePath"
            }
        }
    }

    if ([int]$node.tier -eq 0) { continue }

    $blueprintPath = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $node.blueprint_path))
    $capsulePath = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $node.capsule_path))
    if (-not (Test-Path -LiteralPath $blueprintPath) -or -not (Test-Path -LiteralPath $capsulePath)) { continue }

    $blueprintFront = Get-FrontMatter -Path $blueprintPath
    $capsuleFront = Get-FrontMatter -Path $capsulePath
    if ((Get-ScalarField $blueprintFront 'blueprint_id') -ne $node.id) {
        Add-BlueprintError "Blueprint ID mismatch for $($node.id)"
    }
    if ((Get-ScalarField $capsuleFront 'blueprint_id') -ne $node.id) {
        Add-BlueprintError "Capsule ID mismatch for $($node.id)"
    }
    if ([int](Get-ScalarField $blueprintFront 'revision') -ne [int]$node.revision) {
        Add-BlueprintError "Blueprint revision mismatch for $($node.id)"
    }
    if ([int](Get-ScalarField $capsuleFront 'blueprint_revision') -ne [int]$node.revision) {
        Add-BlueprintError "Stale capsule revision for $($node.id)"
    }
    $parentRevision = [int](Get-ScalarField $blueprintFront 'parent_revision')
    $indexedParentRevision = [int]$nodeById[$node.parent_id].revision
    if ($parentRevision -ne $indexedParentRevision) {
        Add-BlueprintError "Parent revision mismatch for $($node.id): $parentRevision != $indexedParentRevision"
    }

    $frontChildren = @(Get-ListField $blueprintFront 'required_children' | Sort-Object)
    $indexChildren = @(($concreteChildIds + $stubChildIds) | Sort-Object)
    if (($frontChildren -join '|') -ne ($indexChildren -join '|')) {
        Add-BlueprintError "Required-child mismatch for $($node.id)"
    }

    $wordCount = @((Get-Content -Raw -LiteralPath $capsulePath) -split '\s+' | Where-Object { $_ }).Count
    $approxTokens = [math]::Ceiling($wordCount / 0.75)
    if ($approxTokens -gt [int]$node.context_budget_tokens) {
        Add-BlueprintError "Capsule budget exceeded for $($node.id): approximately $approxTokens > $($node.context_budget_tokens) tokens"
    }
}

$rootNode = $roots[0]
if ($null -ne $rootNode -and ($index.PSObject.Properties.Name -contains 'human_flourishing_constitution')) {
    $constitutionPath = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $index.human_flourishing_constitution))
    if (-not (Test-Path -LiteralPath $constitutionPath)) {
        Add-BlueprintError "Missing Human-Flourishing Constitution: $($index.human_flourishing_constitution)"
    }
    $rootCapsulePath = [System.IO.Path]::GetFullPath((Join-Path $blueprintRoot $rootNode.capsule_path))
    if (Test-Path -LiteralPath $rootCapsulePath) {
        $rootFront = Get-FrontMatter -Path $rootCapsulePath
        if ([int](Get-ScalarField $rootFront 'source_revision') -ne [int]$rootNode.revision) {
            Add-BlueprintError "Root capsule source revision does not match root index revision"
        }
    }
}

$markdownFiles = @(
    Get-Item -LiteralPath (Join-Path $projectRoot 'PROJECT_BLUEPRINT.md')
    Get-Item -LiteralPath (Join-Path $projectRoot 'AGENTS.md')
    Get-ChildItem -LiteralPath $blueprintRoot -Recurse -File -Filter '*.md'
)

foreach ($file in $markdownFiles) {
    $raw = Get-Content -Raw -LiteralPath $file.FullName
    foreach ($link in [regex]::Matches($raw, '\[[^\]]*\]\((?<target>[^)]+)\)')) {
        $target = $link.Groups['target'].Value
        if ($target -match '^(https?://|#|[A-Za-z]:[\\/]|<)') { continue }
        $relativeTarget = ($target -split '#')[0]
        if ([string]::IsNullOrWhiteSpace($relativeTarget)) { continue }
        $resolvedTarget = [System.IO.Path]::GetFullPath((Join-Path $file.DirectoryName $relativeTarget))
        if (-not (Test-Path -LiteralPath $resolvedTarget)) {
            Add-BlueprintError "Broken link in $($file.FullName): $target"
        }
    }
}

if ($errors.Count -gt 0) {
    Write-Host "Blueprint validation FAILED with $($errors.Count) error(s):"
    foreach ($validationError in $errors) { Write-Host "- $validationError" }
    exit 1
}

Write-Host "Blueprint validation PASS: $($index.nodes.Count) concrete nodes, $($declaredIds.Count) declared child nodes, $($markdownFiles.Count) Markdown files."
exit 0
