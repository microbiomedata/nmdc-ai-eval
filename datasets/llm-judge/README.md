# Submission Suggestor LLM-Judge Eval

Fetches metadata submissions from the authenticated NMDC submission API,
passes each complete submission document through the production
`nmdc-metadata-suggestor-ai-tool` recommendation pipeline, then uses a
separate LLM call to score the result against a configurable rubric.

## Setup

Set `REFRESH_TOKEN` in `.env` or the shell. The API base defaults to
`https://data.microbiomedata.org`; set `SERVER_API_URL` to use another
environment. The client exchanges the refresh token at `/auth/refresh`, then
sends the returned access token as the bearer credential on submission API
requests. Since access tokens are valid for 24 hours, the client refreshes
them automatically before expiry during long runs. Configure the LLM
credentials as described in the top-level
[authentication guide](../../docs/auth.md).

Add one submission portal ID per line to [`submission-ids.txt`](submission-ids.txt).
Blank lines and lines beginning with `#` are ignored:

```text
submission-id-1
submission-id-2
```

The [`judge-rubric.yaml`](judge-rubric.yaml) contains the expert-defined
Accuracy, Factuality, Relevancy, Completeness, and Coherence metrics with their
1–4 anchors and the overall grading scale. Completeness is excluded from the
mean score when it is null for an abstention.

The suggestor uses the production `LLMClient` from
`nmdc-metadata-suggestor-ai-tool`. Its supported providers are `gcp`, `pnnl`,
and `cborg`. By default, the runner selects the only provider whose credentials
are configured; if credentials for more than one provider are present, select
one explicitly. When `--suggestor-model` is omitted, the suggestor package
chooses its default model for that provider.

## Run

```bash
just eval-llm-judge
just eval-llm-judge --suggestor-provider pnnl --suggestor-model gpt-4o --judge-model gpt-4o-mini
just eval-llm-judge --ids-file /tmp/llm-judge-test-ids.txt --debug
```

`--debug` prints bounded API request/response details to stderr, including
response JSON useful for diagnosing unexpected API response shapes. It redacts
token and authorization fields, but submission metadata may contain sensitive
information; only enable it when appropriate and avoid sharing the output
without reviewing it.

The runner uses the production recommendation pipeline for submission parsing
and context construction, including the pipeline's configured DOI enrichment.
The API may return the submission document directly rather than inside a
`metadata_submission` property; the runner normalizes that response shape for
the suggestor pipeline.
Use `--suggestor-provider gcp`, `pnnl`, or `cborg` to select a provider
explicitly, or `auto` to request credential-based detection. The repository's
`CBORG_API_KEY` setting is passed to the suggestor package under its expected
`CBORG_KEY` name without logging the key.

It writes timestamped JSON results under `pipeline-results/`; the output
contains each submission ID, the suggestor response, per-metric judge scores
and rationales, and the mean score. Raw API submission documents are not
written to the results file. Each submission is judged with a newly obtained
model instance and a single prompt containing only that submission's parsed
context, suggestor context messages, and response; prior scoring results or
conversation history are not included.
