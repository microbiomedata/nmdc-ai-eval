# Full-workflow env triad eval (three NMDC studies)

Model `gemini-2.5-flash` via `gcp`, chunk size 25, run 20261015-111827. Reference: the triad stored in NMDC. Cells: mean ontology score (exact-match rate, n scored).

## All studies pooled

| Slot | samples-only | full-workflow |
|---|---|---|
| env_broad_scale | 0.892 (0.499, n=347) | 0.884 (0.354, n=347) |
| env_local_scale | 0.532 (0.141, n=347) | 0.537 (0.150, n=347) |
| env_medium | 0.839 (0.530, n=347) | 0.752 (0.426, n=347) |

## Per study

| Study | Slot | samples-only | full-workflow |
|---|---|---|---|
| 1000-soils | env_broad_scale | 0.745 (0.029, n=139) | 0.742 (0.029, n=139) |
| 1000-soils | env_local_scale | 0.536 (0.223, n=139) | 0.557 (0.245, n=139) |
| 1000-soils | env_medium | 0.934 (0.338, n=139) | 0.891 (0.374, n=139) |
| geodes-lakes | env_broad_scale | 1.000 (1.000, n=119) | 1.000 (1.000, n=119) |
| geodes-lakes | env_local_scale | 0.576 (0.151, n=119) | 0.576 (0.151, n=119) |
| geodes-lakes | env_medium | 0.861 (0.723, n=119) | 0.735 (0.471, n=119) |
| crop-rhizosphere | env_broad_scale | 0.978 (0.562, n=89) | 0.950 (0.000, n=89) |
| crop-rhizosphere | env_local_scale | 0.466 (0.000, n=89) | 0.456 (0.000, n=89) |
| crop-rhizosphere | env_medium | 0.658 (0.573, n=89) | 0.560 (0.449, n=89) |

## 1000-soils (`nmdc:sty-11-28tm5d36`, soil, 139 biosamples)

### Coverage

| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |
|---|---|---|---|---|---|---|
| samples-only | 139/139 | 417 | 0 | 0 | 0 | 362s |
| full-workflow | 139/139 | 417 | 0 | 0 | 0 | 437s |

### DOI retrieval (full-workflow arm)

| DOI | provider | context chars | PDF links |
|---|---|---|---|
| `10.5281/zenodo.7406532` | zenodo | 2,959 | 0 |
| `10.1101/2022.12.12.520098` | None | 1,794 | 0 |
| `10.1016/j.apsoil.2025.106110` | None | 1,126 | 0 |
| `10.1029/2024GL113091` | None | 1,062 | 0 |
| `10.46936/intm.proj.2021.60141/60000423` | emsl | 834 | 0 |
| `10.46936/10.25585/60008461` | jgi | 18 | 0 |

### Relationship to the NMDC triad

| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |
|---|---|---|---|---|---|---|
| env_broad_scale | samples-only | 4 | 81 | 4 | 50 | 0 |
| env_broad_scale | full-workflow | 4 | 83 | 2 | 50 | 0 |
| env_local_scale | samples-only | 31 | 0 | 0 | 108 | 0 |
| env_local_scale | full-workflow | 34 | 0 | 4 | 101 | 0 |
| env_medium | samples-only | 47 | 50 | 34 | 8 | 0 |
| env_medium | full-workflow | 52 | 31 | 34 | 22 | 0 |

### Generic vs specific reference triads (72 samples with `terrestrial biome` as the stored env_broad_scale)

| Slot | Reference style | samples-only | full-workflow |
|---|---|---|---|
| env_broad_scale | generic | 0.867 (0.000, n=72) | 0.890 (0.000, n=72) |
| env_broad_scale | specific | 0.613 (0.060, n=67) | 0.583 (0.060, n=67) |
| env_local_scale | generic | 0.536 (0.264, n=72) | 0.540 (0.181, n=72) |
| env_local_scale | specific | 0.536 (0.179, n=67) | 0.575 (0.313, n=67) |
| env_medium | generic | 0.964 (0.306, n=72) | 0.977 (0.569, n=72) |
| env_medium | specific | 0.902 (0.373, n=67) | 0.798 (0.164, n=67) |

### What each arm suggested

- **samples-only / env_broad_scale**: `grassland biome [ENVO:01000177]` ×24, `broadleaf forest biome [ENVO:01000197]` ×20, `cropland biome [ENVO:01000245]` ×15, `shrubland biome [ENVO:01000176]` ×14, `mixed forest biome [ENVO:01000198]` ×13
- **samples-only / env_local_scale**: `area of deciduous forest [ENVO:01000816]` ×29, `field [ENVO:01000352]` ×29, `prairie [ENVO:00000260]` ×20, `agricultural field [ENVO:00000114]` ×12, `area of evergreen forest [ENVO:01000843]` ×8
- **samples-only / env_medium**: `soil [ENVO:00001998]` ×56, `forest soil [ENVO:00002261]` ×40, `grassland soil [ENVO:00005750]` ×20, `agricultural soil [ENVO:00002259]` ×15, `dry soil [ENVO:00005748]` ×2
- **full-workflow / env_broad_scale**: `broadleaf forest biome [ENVO:01000197]` ×24, `grassland biome [ENVO:01000177]` ×18, `cropland biome [ENVO:01000245]` ×15, `forest biome [ENVO:01000174]` ×12, `mixed forest biome [ENVO:01000198]` ×10
- **full-workflow / env_local_scale**: `field [ENVO:01000352]` ×35, `prairie [ENVO:00000260]` ×22, `agricultural field [ENVO:00000114]` ×15, `woodland area [ENVO:00000109]` ×13, `area of deciduous forest [ENVO:01000816]` ×13
- **full-workflow / env_medium**: `soil [ENVO:00001998]` ×75, `surface soil [ENVO:02000059]` ×30, `forest soil [ENVO:00002261]` ×18, `grassland soil [ENVO:00005750]` ×7, `agricultural soil [ENVO:00002259]` ×5

### Change from samples-only to full-workflow (graded by ENVO proximity)

| Slot | changed | toward NMDC | away from NMDC |
|---|---|---|---|
| env_broad_scale | 56/139 | 20 | 12 |
| env_local_scale | 64/139 | 14 | 7 |
| env_medium | 68/139 | 28 | 29 |

## geodes-lakes (`nmdc:sty-11-fkbnah04`, freshwater, 119 biosamples)

### Coverage

| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |
|---|---|---|---|---|---|---|
| samples-only | 119/119 | 357 | 0 | 0 | 0 | 268s |
| full-workflow | 119/119 | 357 | 0 | 0 | 0 | 298s |

### DOI retrieval (full-workflow arm)

| DOI | provider | context chars | PDF links |
|---|---|---|---|
| `10.25345/C5QB9VG8C` | massive | 923 | 0 |
| `10.1002/lno.11306` | None | 981 | 0 |
| `10.46936/10.25585/60000947` | jgi | 21 | 0 |

### Relationship to the NMDC triad

| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |
|---|---|---|---|---|---|---|
| env_broad_scale | samples-only | 119 | 0 | 0 | 0 | 0 |
| env_broad_scale | full-workflow | 119 | 0 | 0 | 0 | 0 |
| env_local_scale | samples-only | 18 | 0 | 0 | 101 | 0 |
| env_local_scale | full-workflow | 18 | 0 | 0 | 101 | 0 |
| env_medium | samples-only | 86 | 0 | 0 | 33 | 0 |
| env_medium | full-workflow | 56 | 0 | 0 | 63 | 0 |

### What each arm suggested

- **samples-only / env_broad_scale**: `freshwater lake biome [ENVO:01000252]` ×119
- **samples-only / env_local_scale**: `freshwater lake [ENVO:00000021]` ×97, `epilimnion [ENVO:00002131]` ×18, `hypolimnion [ENVO:00002130]` ×4
- **samples-only / env_medium**: `lake water [ENVO:04000007]` ×86, `eutrophic water [ENVO:00002224]` ×17, `oligotrophic water [ENVO:00002223]` ×16
- **full-workflow / env_broad_scale**: `freshwater lake biome [ENVO:01000252]` ×119
- **full-workflow / env_local_scale**: `freshwater lake [ENVO:00000021]` ×97, `epilimnion [ENVO:00002131]` ×18, `hypolimnion [ENVO:00002130]` ×4
- **full-workflow / env_medium**: `lake water [ENVO:04000007]` ×56, `oligotrophic water [ENVO:00002223]` ×33, `eutrophic water [ENVO:00002224]` ×30

### Change from samples-only to full-workflow (graded by ENVO proximity)

| Slot | changed | toward NMDC | away from NMDC |
|---|---|---|---|
| env_broad_scale | 0/119 | 0 | 0 |
| env_local_scale | 0/119 | 0 | 0 |
| env_medium | 36/119 | 3 | 33 |

## crop-rhizosphere (`nmdc:sty-11-prtb4s28`, plant-associated, 89 biosamples)

### Coverage

| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |
|---|---|---|---|---|---|---|
| samples-only | 89/89 | 267 | 0 | 0 | 0 | 204s |
| full-workflow | 89/89 | 267 | 0 | 0 | 0 | 306s |

### DOI retrieval (full-workflow arm)

| DOI | provider | context chars | PDF links |
|---|---|---|---|
| `10.25345/C5JS9HJ9J` | massive | 1,049 | 0 |
| `10.25345/C58C9RF3K` | massive | 1,049 | 0 |
| `10.46936/10.25585/60000944` | jgi | 29 | 0 |

### Relationship to the NMDC triad

| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |
|---|---|---|---|---|---|---|
| env_broad_scale | samples-only | 50 | 39 | 0 | 0 | 0 |
| env_broad_scale | full-workflow | 0 | 89 | 0 | 0 | 0 |
| env_local_scale | samples-only | 0 | 0 | 0 | 89 | 0 |
| env_local_scale | full-workflow | 0 | 0 | 0 | 89 | 0 |
| env_medium | samples-only | 51 | 0 | 0 | 38 | 0 |
| env_medium | full-workflow | 40 | 0 | 0 | 49 | 0 |

### What each arm suggested

- **samples-only / env_broad_scale**: `terrestrial biome [ENVO:00000446]` ×50, `anthropogenic terrestrial biome [ENVO:01000219]` ×39
- **samples-only / env_local_scale**: `rhizosphere [ENVO:00005801]` ×65, `agricultural field [ENVO:00000114]` ×14, `root [PO:0009005]` ×10
- **samples-only / env_medium**: `environmental material [ENVO:00010483]` ×38, `root [PO:0009005]` ×37, `soil [ENVO:00001998]` ×14
- **full-workflow / env_broad_scale**: `anthropogenic terrestrial biome [ENVO:01000219]` ×89
- **full-workflow / env_local_scale**: `rhizosphere [ENVO:00005801]` ×62, `agricultural field [ENVO:00000114]` ×14, `root [PO:0009005]` ×13
- **full-workflow / env_medium**: `environmental material [ENVO:00010483]` ×49, `root [PO:0009005]` ×26, `soil [ENVO:00001998]` ×14

### Change from samples-only to full-workflow (graded by ENVO proximity)

| Slot | changed | toward NMDC | away from NMDC |
|---|---|---|---|
| env_broad_scale | 50/89 | 0 | 50 |
| env_local_scale | 23/89 | 0 | 0 |
| env_medium | 11/89 | 0 | 11 |
