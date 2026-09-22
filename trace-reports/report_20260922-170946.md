# Production trace report

84 traces, 2026-07-14 to 2026-09-22. No model was called to produce this report.

## What the agent did

| Measure | Value |
|---|---|
| traces carrying a run health block | 25 of 84 |
| of those, hit at least one permission denial | 12 |
| of those, reported `completed` anyway | 12 |
| most denials in a single run | 205 |
| total cost recorded | $25.3307 |
| traces whose metadata names a different model than the AGENT span | 21 |

## Whether the env triad values are well formed

Reference free. Each check needs ENVO, not a curated answer.

| Check | Count | Denominator |
|---|---|---|
| values suggested | 918 | |
| parse as `label [CURIE]` | 918 | 918 |
| CURIE resolves in ENVO | 918 | 918 |
| label matches ENVO's label | 911 | 918 |
| all three | 911 | 918 |

Non-ENVO prefixes seen: none

## Output shapes

| Shape | Traces |
|---|---|
| none | 34 |
| LLMOutput | 32 |
| MetadataMapperOutput | 9 |
| not-a-dict:list | 7 |
| not-a-dict:str | 2 |

## Trace names

| Name | Traces |
|---|---|
| agentic | 50 |
| metadata_mapper_agentic | 12 |
| <none> | 7 |
| env-triad-fixture-run | 7 |
| ClaudeAgentSDK.ClaudeSDKClient.receive_response | 3 |
| pr139-restricted-bypass | 1 |
| pr139-cc-by-bypass | 1 |
| pr139-restricted-default | 1 |
| pr139-cc-by-default | 1 |
| ClaudeAgentSDK.query | 1 |
