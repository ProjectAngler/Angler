---
evidence_id: ANG-EVID-CR0-SAFETY-DESIGN-001
status: accepted_for_bootstrap_design
created_at: 2026-08-25
scope: Construction Release 0 safety design only
authorization_kind: BOOTSTRAP_WORK
gate: ANG-GATE-CONSTRUCTION-RELEASE-0-001@1
assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001
---

# CR0 safety-design evidence manifest

## Claim boundary

This evidence supports the CR0 bootstrap-design decision only. It does not prove OS-level containment, learner/model safety, a human-flourishing-gate pass, SAFETY Tier-1 completion, Slice 00, M0, or any scientific/technical result.

## Validation result

Command:

```powershell
& 'docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1'
```

Result:

```text
PASS - CR0 safety design, bootstrap non-equivalence, authority, permission ceilings, assessment, and rollback hash validated.
```

The test is static design validation. It verifies required files/semantics and independently hashes the rollback archive; it is not a sandbox penetration test.

## Rollback identity

- Path: `work/pre-construction-release-0-20260825.zip`
- SHA-256: `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`
- Verification: PASS on 2026-08-25

## Artifact hashes

SHA-256 values were calculated after the passing safety-specific test:

```text
A9A5D033E14B4AA1A9C02170462837C777541D5F9058544E699C838516897D3E  children/threat-model/BLUEPRINT.md
DB5CB7563A0D24409C0424E10FD49E49BE654468602B02826701C70482EA70D1  children/threat-model/CAPSULE.md
6DEF4E7C008DD3A83516ED9DE26CE851566AE3E190EDF33FE04C0BAB7FE6B155  children/threat-model/STATUS.md
190663448E4282ED050F7D83627E89E86F67D92B61D2153BFE88EDED2F13D84A  children/threat-model/gates/ANG-GATE-THREAT-MODEL-001.md
F0E9C8BE1D868381636B25B32A1CB8C04C48B90EF5026A0B657AF75C56E168F1  children/human-authority/BLUEPRINT.md
E6FDA8E39F245DBD6038C3FCE4336497CCEE61E27D666607DD1E8EBDB1115E9C  children/human-authority/CAPSULE.md
76F05A972A83FC4F291985F0EA072FF11462F6717FC41A2FF19E0401485AC2E7  children/human-authority/STATUS.md
A0D998E96CB927C28021490262DFA54649FEFE31F5C7B01B48AB83540E511520  children/human-authority/gates/ANG-GATE-HUMAN-AUTHORITY-001.md
15F02C2E2BCA86FC9D8E9ACB2806B10742B51FE013E1C122AE824A76635B40ED  children/permissions/BLUEPRINT.md
CDC0A2831B29E49BB01E8DD833B5F3077181C9819428624CD8B751CBA2AF659A  children/permissions/CAPSULE.md
7614173B998EFFFBC0566F1CBE1B3A36C8E37E8CA434887D4E82CC6353AAC664  children/permissions/STATUS.md
53806853DE036CD9DA6C806F9B7A9F025CABD20366FEBAAD4A36AC4896D5FB85  children/permissions/gates/ANG-GATE-PERMISSIONS-001.md
23D04D544208C7273BA6C7860CC788CDD81640C8DD8236FFD1FED1F2D77495C6  policies/ANG-POL-LOCAL-SCAFFOLD-001.md
181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC  assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md
7A060520A1AAECC02866B2C72FC7F355C0341FD98A9C549B5DAD697C32A9079B  contracts/ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001.md
07776545979E1A1B3AE0787005E6782013F3127924F3D4FF6479C9470CBB2E2A  gates/ANG-GATE-HUMAN-FLOURISHING-001.md
B768F7669241A0C3432E95E0DDB900AE5A007B2369AB3E4D073F473434DE8EEB  gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md
7D8ED013112C3560136593666691E12C796BCB65D72FAA30FDC356AC01E5A19D  gates/ANG-GATE-SAFETY-DESIGN-001.md
41735619A6C06A9360E3809B027F1FA4015CF1D3B05B69BB3E467429FC7C644A  tests/Test-Cr0SafetyDesign.ps1
43F912AF5305AFF59F50E23F5058238BA4D7CB5DDF1B1C6592CC25A89C8AEEE6  CAPSULE.md
E2C4A5252DA1811364E2BD079BBEF4B6C005D7AF838DC16CFF8C538978311DBE  STATUS.md
632C80A03898CE2A55AE47E386690DABCC4ED20B4D3A1822CE9EACCF8D5E4C25  HANDOFF.md
```

Paths above are relative to `docs/blueprints/branches/safety/` except the rollback archive.

## Root validation note

The repository-level blueprint validator initially encountered an outside-SAFETY parser defect. After that defect was repaired concurrently, it ran and reported only two unrelated EVIDENCE revision/capsule mismatches. This evidence does not claim a repository-level pass; it records that no SAFETY-specific validation error was reported in the subsequent run.

## Decision

Accept the safety design as evidence for the separate CR0 bootstrap gate. Keep `ANG-GATE-HUMAN-FLOURISHING-001`, `ANG-GATE-SAFETY-DESIGN-001`, Slice 00, and M0 open.
