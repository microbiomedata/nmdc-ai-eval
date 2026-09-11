# Supplement-driven env triad eval (phyllosphere)

Study `nmdc:sty-11-e4yb9z58`, DOI `10.1038/s41467-023-36515-y`, model `gemini-2.5-flash` via `gcp`, 192 biosamples, chunk size 25, run 20260911-180831.

## Retrieval

- Supplements kept: 10 (inlined as text: 2), skipped: 18, 3.6s
- Supplement context: 215,276 characters
- Samples without a supplement row: 0

## Coverage

| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |
|---|---|---|---|---|---|---|
| publication | 192/192 | 576 | 0 | 0 | 0 | 364s |
| publication+supplements | 191/192 | 627 | 0 | 0 | 0 | 825s |

## Ontology score per slot (mean; exact-match rate in parentheses)

Scored samples are those answered by the arm *and* holding a reference term for the slot.

| Slot | Reference | publication | publication+supplements |
|---|---|---|---|
| env_broad_scale | supplement | 0.900 (0.000, n=192) | 0.974 (0.744, n=191) |
| env_broad_scale | nmdc | 0.500 (0.000, n=192) | 0.500 (0.000, n=191) |
| env_local_scale | supplement | 0.591 (0.260, n=192) | 1.000 (1.000, n=191) |
| env_local_scale | nmdc | 0.461 (0.000, n=192) | 0.500 (0.000, n=191) |
| env_medium | supplement | 0.500 (0.000, n=192) | 0.570 (0.351, n=191) |
| env_medium | nmdc | 0.500 (0.000, n=192) | 0.395 (0.000, n=191) |

## Relationship to the supplement reference

| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |
|---|---|---|---|---|---|---|
| env_broad_scale | publication | 0 | 192 | 0 | 0 | 0 |
| env_broad_scale | publication+supplements | 142 | 49 | 0 | 0 | 0 |
| env_local_scale | publication | 50 | 0 | 0 | 142 | 0 |
| env_local_scale | publication+supplements | 191 | 0 | 0 | 0 | 0 |
| env_medium | publication | 0 | 0 | 0 | 192 | 0 |
| env_medium | publication+supplements | 67 | 0 | 0 | 124 | 0 |

## What each arm suggested

### publication

- **env_broad_scale**: `cropland biome [ENVO:01000245]` ×192
  - gate: submission_enum 192; accepted 192
- **env_local_scale**: `crop canopy [ENVO:01001241]` ×117, `area of cropland [ENVO:01000892]` ×50, `agricultural field [ENVO:00000114]` ×25
  - gate: submission_enum 167, envo_expansion 25; accepted 192
- **env_medium**: `leaf [PO:0025034]` ×192
  - gate: submission_enum 192; accepted 192

### publication+supplements

- **env_broad_scale**: `terrestrial biome [ENVO:00000446]` ×142, `cropland biome [ENVO:01000245]` ×49
  - gate: submission_enum 191; accepted 191
- **env_local_scale**: `area of cropland [ENVO:01000892]` ×191
  - gate: submission_enum 191; accepted 191
- **env_medium**: `leaf [PO:0025034]` ×124, `plant matter [ENVO:01001121]` ×67
  - gate: envo_expansion 67, submission_enum 124; accepted 191

## Change from adding supplements (graded by ENVO proximity)

| Slot | changed | toward supplement | away from supplement | toward NMDC | away from NMDC |
|---|---|---|---|---|---|
| env_broad_scale | 142/191 | 142 | 0 | 0 | 0 |
| env_local_scale | 141/191 | 141 | 0 | 0 | 0 |
| env_medium | 67/191 | 67 | 0 | 0 | 0 |

Transitions for env_broad_scale:

- 142× `cropland biome [ENVO:01000245] -> terrestrial biome [ENVO:00000446]`

Transitions for env_local_scale:

- 116× `crop canopy [ENVO:01001241] -> area of cropland [ENVO:01000892]`
- 25× `agricultural field [ENVO:00000114] -> area of cropland [ENVO:01000892]`

Transitions for env_medium:

- 67× `leaf [PO:0025034] -> plant matter [ENVO:01001121]`
