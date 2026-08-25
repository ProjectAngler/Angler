# CR0 Evidence-Schemas scaffold acceptance addendum

This append-only addendum records the historical CR0 disposition without modifying the original executor handoff, receipts, decision, contracts, tests, or implementation. It is not a normal technical, Human-Flourishing, Slice-00, M0, scientific, model, or deployment pass.

## Bound authority and disposition

- Accepted commit: `903f9b9d5e58818d774604dbd6f4d89b2b4544e0`
- Independent disposition: `SCAFFOLD_ACCEPTED`
- Decision SHA-256: `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`
- Leaf SHA-256: `5CF4A9DE5B8FAD71BCC27B9CF71ED66BCF8183AE4A71655805503FD832C53289`
- Gate SHA-256: `A883FE87B366716A54B412FB49F0FAF280A90D2432950D239B46507F389548F5`
- Baseline SHA-256: `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F`
- Execution Manifest v2 SHA-256: `802D1525C96339902C7D44E3E1C61CD698742532D4E4A05F80700AE9DC13E5D2`
- Test receipt SHA-256: `897D57E623B295F52EC10A11E7873630DA20E2956248F59B7F097E569A7C3E53`
- Effect receipt SHA-256: `9808AFC8BF19EE496D64E30C01BC494BB8F9C1C3459672BBFCA2C1B43639C893`
- Original executor handoff SHA-256: `017969B6AC9B91D277E4F5323D2F9AD8275D4547E81D3085C42A286CE67FC855`

## Exact implementation identities

| Path | SHA-256 |
|---|---|
| `src/angler/episodes/__init__.py` | `62E298AD19F52B6A620C4D62B67116B2A86B6294C68B7CD2A0B9C60BC1D6A0FE` |
| `src/angler/episodes/canonical.py` | `E0E1B85D4C00CED3BB90917A4C37D3D3950ACAC9A1DB7FAA47A6AA897E869A72` |
| `src/angler/episodes/schema_validation.py` | `74F9B5F3FFE248D8E0B667AF1774BE92E29F55F89E87C2983C4431017611AF3D` |
| `src/angler/episodes/visibility.py` | `DBD94B64D196665048479AF4EA1B902525B2E591601B01BEB51D5C00D7F290D3` |
| `src/angler/episodes/schemas/evidence-envelope.v1.json` | `E8762C3576D4DFCAF46B833905A589551089A5B606054D273EFEC31B887D6CB6` |
| `src/angler/episodes/schemas/episode.v1.json` | `B9925BB0B37B43AC508535DA9635A40766C63FBBD36BEA2D04168EEAD7D79DBE` |
| `src/angler/episodes/schemas/experiment-manifest.v1.json` | `4002E8FA87BEE91365A6E3CA926EEC758F7F53992DBAB2D7546A6CCA3CEFA9DF` |
| `tests/unit/evidence/test_evidence_schemas.py` | `12C5014F6168D36A3D16B12B1103AACB5A2DE8DD3056D5C2325C5229844CAD5D` |
| `tests/fixtures/evidence_schemas/valid-envelope.json` | `B412BC9BD060864D2154508B4ABFF5E8ADCE6D4EC4FAA1C384175EE270CB86FA` |
| `tests/fixtures/evidence_schemas/valid-episode.json` | `42828EB83ACD120D1A1E9170A928A49A28006777ED4D4D8B2A3A0E97B35E6069` |
| `tests/fixtures/evidence_schemas/valid-experiment-manifest.json` | `F772992D4C28DA7A323EC10DFD689DAD0ECB04297C2F9F99DD09F9526161FA65` |
| `tests/fixtures/evidence_schemas/invalid-cases.json` | `F2AC89298CAB14BD0A7D3A2AB34FCE1CED991FF78431475EA5A5B1F767053B8A` |
| `tests/fixtures/evidence_schemas/visibility-matrix.json` | `370783E5661A7F3B00CED93D08E1466E6704946221FE3E5680D1854AE42207C7` |
| `tests/fixtures/evidence_schemas/sealed-commitment-cases.json` | `91137160496B4CA8E49D3A1110ED3C429E0DDE9B8632E4B046251B62B45FF409` |

## Non-equivalence and continuation

The CR0 disposition accepts only these exact scaffold bytes. The normal `ANG-GATE-EVIDENCE-SCHEMAS-001` and `ANG-GATE-HUMAN-FLOURISHING-001` remain `NOT_RUN`; Slice 00 and M0 remain `NOT_PASSED`. The leaf is historical and must not be rerun. ARTIFACT-LINEAGE remains blocked by the normal gate. Any successor executable work requires a new manifest and fresh authority.
