# Submission Suggestor LLM-Judge Eval

Runs the metadata suggestor and scores its suggestions with an LLM judge.

## Inputs

- Add one submission portal ID per line to [`submission-ids.txt`](submission-ids.txt). Blank lines and lines beginning with `#` are ignored.
- The evaluation rubric is in [`judge-rubric.yaml`](judge-rubric.yaml).
- Set `REFRESH_TOKEN` and the credentials for your LLM provider in `.env` or the shell. See the [authentication guide](../../docs/auth.md).
- Optionally set `SERVER_API_URL` to use a different submission API.

## Run

From the repository root:

```bash
just eval-llm-judge
```

Choose the suggestor and judge models or use a different input file:

```bash
just eval-llm-judge --suggestor-provider pnnl --suggestor-model gpt-4o --judge-model gpt-4o-mini
just eval-llm-judge --ids-file /path/to/ids.txt --rubric-file /path/to/rubric.yaml
just eval-llm-judge --version 1
```

The suggestor provider can be `gcp`, `pnnl`, or `cborg`. If omitted, the runner detects a provider when exactly one provider's credentials are configured. The suggestor model defaults to the provider's default; the judge model defaults to `gpt-5.2`.

## Output

Timestamped JSON files are written to `pipeline-results/00/` by default. Choose a different pipeline version with `--version`; for example, `--version 1` writes to `pipeline-results/01/`. Each file contains run metadata and a `results` array with one entry per submission. An entry includes the suggestor response, per-suggestion judge scores and findings, and a derived `mean_score`:

```json
{
  "submission_id": "submission-id-1",
  "suggestor_response": {},
  "suggestions": [],
  "mean_score": 3.5
}
```

The mean is calculated from non-null metric scores; completeness is null for abstentions and excluded from the mean. Raw submission documents are not included in the output.
