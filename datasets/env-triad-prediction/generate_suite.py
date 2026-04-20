#!/usr/bin/env python3
"""Generate an llm-matrix suite YAML for env triad prediction.

Reads ground_truth.yaml and fetches full submission documents from MongoDB
(nmdc_data_dev.nmdc_submissions). Uses the production suggestor's own system
prompt, submission field extraction, and schema context — the only difference
from the real pipeline is the LLM provider (llm-matrix vs Vertex/PNNL).

For testing the real end-to-end pipeline, see run_pipeline_eval.py.

Usage:
    python generate_suite.py                    # defaults
    python generate_suite.py --mongo-uri ...    # custom MongoDB
"""

import argparse
from pathlib import Path

import yaml
from nmdc_api_utilities.study_search import StudySearch

HERE = Path(__file__).parent
MODELS_YAML = HERE.parent / "models.yaml"
OUTPUT_YAML = HERE / "env-triad-suite.yaml"


def load_models() -> list[str]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def get_bioscales_study_example() -> list[dict]:
    """Get all biosamples associated with the Bioscales study."""
    study_id = "nmdc:sty-11-r2h77870"
    # call NMDC API
    search = StudySearch()
    # biosample_search = BiosampleSearch()
    study = search.get_record_by_id(collection_id=study_id)
    # get the linked biosamples
    biosamples = search.get_linked_instances(ids=[study_id], types="nmdc:Biosample", hydrate=True, max_page_size=1999)
    # # get the ids of the biosamples
    # biosample_ids = [biosample["id"] for biosample in biosamples]
    # # get the full biosample records
    # biosamples = biosample_search.get_batch_records(id_list=biosample_ids, search_field="id")

    return {"study": study, "biosamples": biosamples}


def format_prompt(biosample: dict, study: dict) -> str:
    """Build a prompt from the biosample, study, and SchemaContextBuilder."""
    copy = biosample.copy()  # avoid mutating the original
    # remove env triad from biosample
    copy.pop("env_broad_scale", None)
    copy.pop("env_local_scale", None)
    copy.pop("env_medium", None)
    prompt_parts = []
    prompt_parts.append(f"--- Study Metadata ---{study}")
    prompt_parts.append(f"--- Biosample Metadata ---{copy}")
    prompt_parts.append("Suggest env triad values for the biosample")

    return "\n".join(prompt_parts)


def make_ideal(env_broad_scale, env_local_scale, env_medium) -> str:
    """JSON-formatted ideal answer matching the production output schema."""
    import json

    fields = [
        {
            "field_name": "env_broad_scale",
            "reason": "",
            "value": f"{env_broad_scale.get('term').get('name')} [{env_broad_scale.get('term').get('id')}]",
        },
        {
            "field_name": "env_local_scale",
            "reason": "",
            "value": f"{env_local_scale.get('term').get('name')} [{env_local_scale.get('term').get('id')}]",
        },
        {
            "field_name": "env_medium",
            "reason": "",
            "value": f"{env_medium.get('term').get('name')} [{env_medium.get('term').get('id')}]",
        },
    ]
    return json.dumps({"metadata_fields": fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate env-triad llm-matrix suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_YAML,
        help="Output suite YAML path",
    )
    args = parser.parse_args()

    models = load_models()
    study_example = get_bioscales_study_example()
    cases = []
    for biosample in study_example["biosamples"]:
        cases.append(
            {
                "input": format_prompt(biosample, study_example["study"]),
                "ideal": make_ideal(
                    biosample.get("env_broad_scale"),
                    biosample.get("env_local_scale"),
                    biosample.get("env_medium"),
                ),
                "tags": ["value_prediction"],
            }
        )

    suite = {
        "name": "Env Triad Prediction Test Suite",
        "template": "env_triad_prediction",
        "templates": {
            "env_triad_prediction": {
                "system": """Use the provided information to suggest values for the
                following three metadata fields: env_broad_scale, env_local_scale, and env_medium.
          - env_broad_scale: This field should capture the broad environmental
          category of the sample (e.g., aquatic, terrestrial, host-associated).
          - env_local_scale: This field should capture the more specific local
          environment of the sample (e.g., sediment, water column, rhizosphere).
          - env_medium: This field should capture the medium in which the
          sample was collected (e.g., soil, water, air).
          The values for these fields should be chosen from the enumerations in the
          NMDC schema and should be based on the content of the information provided.

          Output schema:
          ```json
          {
              "metadata_fields": [
                  {
                      "field_name": "env_broad_scale",
                      "reason": "Reason for choosing this field based on the provided information.",
                      "value": ""
                  },
                  {
                      "field_name": "env_local_scale",
                      "reason": "Reason for choosing this field based on the provided information.",
                      "value": ""
                  },
                  {
                      "field_name": "env_medium",
                      "reason": "Reason for choosing this field based on the provided information.",
                      "value": ""
                  }
              ]
          }```
          Schema context to choose from:
          # AirInterface - Selected Fields\n- **env_broad_scale**\n  In this field, \
report which major environmental system your sample or specimen came from. The systems \
identified should have a coarse spatial grain, to provide the general environmental \
context of where the sampling was done (e.g. were you in the desert or a rainforest?). \
We recommend using subclasses of ENVO's biome class: \
http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel [termID], \
Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel [termID]. \
Example: Annotating a water sample from the photic zone in middle of the Atlantic \
Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: Annotating a \
sample from the Amazon rainforest consider: tropical moist broadleaf forest biome \
[ENVO:01000228]. If needed, request new terms on the ENVO tracker, identified here: \
http://www.obofoundry.org/ontology/envo.html\n  (type: string, pattern: \
`^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_local_scale**\n  \
Report the entity or entities which are in the sample or specimen s local vicinity and \
which you believe have significant causal influences on your sample or specimen. We \
recommend using EnvO terms which are of smaller spatial grain than your entry for \
env_broad_scale. Terms, such as anatomical sites, from other OBO Library ontologies \
which interoperate with EnvO (e.g. UBERON) are accepted in this field. EnvO \
documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# BiofilmInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# BuiltEnvInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# HcrCoresInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# HcrFluidsSwabsInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# HostAssociatedInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[(ENVO:\\d{7,8}|UBERON:\\d{7})\\]$`)\n- \
**env_medium**\n  Report the environmental material(s) immediately surrounding the \
sample or specimen at the time of sampling. We recommend using subclasses of \
'environmental material' (http://purl.obolibrary.org/obo/ENVO_00010483). EnvO \
documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms from \
other OBO ontologies are permissible as long as they reference mass/volume nouns (e.g. \
air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a table \
top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[(ENVO:\\d{7,8}|UBERON:\\d{7})\\]$`)\n\n---\n\n# \
MetagenomeSequencingInterleavedDataInterface - Selected Fields\n\n---\n\n# \
MetagenomeSequencingNonInterleavedDataInterface - Selected Fields\n\n---\n\n# \
MetatranscriptomeSequencingInterleavedDataInterface - Selected Fields\n\n---\n\n# \
MetatranscriptomeSequencingNonInterleavedDataInterface - Selected Fields\n\n---\n\n# \
MiscEnvsInterface - Selected Fields\n- **env_broad_scale**\n  In this field, report \
which major environmental system your sample or specimen came from. The systems \
identified should have a coarse spatial grain, to provide the general environmental \
context of where the sampling was done (e.g. were you in the desert or a rainforest?). \
We recommend using subclasses of ENVO's biome class: \
http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel [termID], \
Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel [termID]. \
Example: Annotating a water sample from the photic zone in middle of the Atlantic \
Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: Annotating a \
sample from the Amazon rainforest consider: tropical moist broadleaf forest biome \
[ENVO:01000228]. If needed, request new terms on the ENVO tracker, identified here: \
http://www.obofoundry.org/ontology/envo.html\n  (type: string, pattern: \
`^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_local_scale**\n  \
Report the entity or entities which are in the sample or specimen s local vicinity and \
which you believe have significant causal influences on your sample or specimen. We \
recommend using EnvO terms which are of smaller spatial grain than your entry for \
env_broad_scale. Terms, such as anatomical sites, from other OBO Library ontologies \
which interoperate with EnvO (e.g. UBERON) are accepted in this field. EnvO \
documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# PlantAssociatedInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n  Allowed values: \
alpine tundra biome [ENVO:01001505], anthropogenic terrestrial biome [ENVO:01000219], \
aquatic biome [ENVO:00002030], broadleaf forest biome [ENVO:01000197], coniferous \
forest biome [ENVO:01000196], cropland biome [ENVO:01000245], estuarine biome \
[ENVO:01000020], flooded grassland biome [ENVO:01000195], flooded savanna biome \
[ENVO:01000190], forest biome [ENVO:01000174], freshwater biome [ENVO:00000873], \
freshwater lake biome [ENVO:01000252], freshwater river biome [ENVO:01000253], \
freshwater stream biome [ENVO:03605008], grassland biome [ENVO:01000177], large \
freshwater lake biome [ENVO:00000891], large river biome [ENVO:00000887], large river \
delta biome [ENVO:00000889], large river headwater biome [ENVO:00000888], mangrove \
biome [ENVO:01000181], marine biome [ENVO:00000447], marine neritic benthic zone biome \
[ENVO:01000025], marine salt marsh biome [ENVO:01000022], mediterranean forest biome \
[ENVO:01000199], mediterranean grassland biome [ENVO:01000224], mediterranean savanna \
biome [ENVO:01000229], mediterranean shrubland biome [ENVO:01000217], mediterranean \
woodland biome [ENVO:01000208], mixed forest biome [ENVO:01000198], montane grassland \
biome [ENVO:01000194], montane savanna biome [ENVO:01000223], montane shrubland biome \
[ENVO:01000216], neritic epipelagic zone biome [ENVO:01000042], neritic mesopelagic \
zone biome [ENVO:01000043], neritic pelagic zone biome [ENVO:01000032], neritic sea \
surface microlayer biome [ENVO:01000041], rangeland biome [ENVO:01000247], savanna \
biome [ENVO:01000178], shrubland biome [ENVO:01000176], small freshwater lake biome \
[ENVO:00000892], small river biome [ENVO:00000890], subpolar coniferous forest biome \
[ENVO:01000250], subtropical broadleaf forest biome [ENVO:01000201], subtropical \
coniferous forest biome [ENVO:01000209], subtropical dry broadleaf forest biome \
[ENVO:01000225], subtropical grassland biome [ENVO:01000191], subtropical moist \
broadleaf forest biome [ENVO:01000226], subtropical savanna biome [ENVO:01000187], \
subtropical shrubland biome [ENVO:01000213], subtropical woodland biome \
[ENVO:01000222], temperate broadleaf forest biome [ENVO:01000202], temperate coniferous \
forest biome [ENVO:01000211], temperate grassland biome [ENVO:01000193], temperate \
mixed forest biome [ENVO:01000212], temperate savanna biome [ENVO:01000189], temperate \
shrubland biome [ENVO:01000215], temperate woodland biome [ENVO:01000221], terrestrial \
biome [ENVO:00000446], tidal mangrove shrubland [ENVO:01001369], tropical broadleaf \
forest biome [ENVO:01000200], tropical coniferous forest biome [ENVO:01000210], \
tropical dry broadleaf forest biome [ENVO:01000227], tropical grassland biome \
[ENVO:01000192], tropical mixed forest biome [ENVO:01001798], tropical moist broadleaf \
forest biome [ENVO:01000228], tropical savanna biome [ENVO:01000188], tropical \
shrubland biome [ENVO:01000214], tropical woodland biome [ENVO:01000220], tundra biome \
[ENVO:01000180], woodland biome [ENVO:01000175], xeric basin biome [ENVO:00000893], \
xeric shrubland biome [ENVO:01000218]\n- **env_local_scale**\n  Report the entity or \
entities which are in the sample or specimen s local vicinity and which you believe \
have significant causal influences on your sample or specimen. We recommend using EnvO \
terms which are of smaller spatial grain than your entry for env_broad_scale. Terms, \
such as anatomical sites, from other OBO Library ontologies which interoperate with \
EnvO (e.g. UBERON) are accepted in this field. EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: \
string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  Allowed values: agricultural terrace \
[ENVO:00000519], alluvial plain [ENVO:00000258], area of barren land [ENVO:01000752], \
area of cropland [ENVO:01000892], area of deciduous forest [ENVO:01000816], area of \
developed open space [ENVO:01000883], area of developed space with high usage intensity \
[ENVO:01000886], area of developed space with low usage intensity [ENVO:01000884], area \
of developed space with medium usage intensity [ENVO:01000885], area of dwarf scrub \
[ENVO:01000861], area of emergent herbaceous wetland [ENVO:01000894], area of evergreen \
forest [ENVO:01000843], area of gramanoid or herbaceous vegetation [ENVO:01000888], \
area of lichen-dominated vegetation [ENVO:01000889], area of mixed forest \
[ENVO:01000855], area of moss-dominated vegetation [ENVO:01000890], area of open water \
[ENVO:01000666], area of perennial ice or snow [ENVO:01000746], area of perennial snow \
[ENVO:01000745], area of perennial water ice [ENVO:01000740], area of scrub \
[ENVO:01000869], area of sedge- and forb-dominated herbaceous vegetation \
[ENVO:01000887], area of woody wetland [ENVO:01000893], beach [ENVO:00000091], \
botanical garden [ENVO:00010624], cliff [ENVO:00000087], coast [ENVO:01000687], crop \
canopy [ENVO:01001241], desert [ENVO:01001357], dune [ENVO:00000170], farm \
[ENVO:00000078], forest floor [ENVO:01001582], garden [ENVO:00000011], greenhouse \
[ENVO:03600087], harbour [ENVO:00000463], herb and fern layer [ENVO:01000337], hill \
[ENVO:00000083], house [ENVO:01000417], island [ENVO:00000098], laboratory facility \
[ENVO:01001406], litter layer [ENVO:01000338], market [ENVO:01000987], mountain \
[ENVO:00000081], oasis [ENVO:01001304], ocean [ENVO:00000015], outcrop [ENVO:01000302], \
plantation [ENVO:00000117], plateau [ENVO:00000182], pond [ENVO:00000033], prairie \
[ENVO:00000260], public park [ENVO:03500002], research facility [ENVO:00000469], river \
bank [ENVO:00000143], river valley [ENVO:00000171], road [ENVO:00000064], sea grass bed \
[ENVO:01000059], shore [ENVO:00000304], shrub layer [ENVO:01000336], submerged bed \
[ENVO:00000501], understory [ENVO:01000335], valley [ENVO:00000100], woodland canopy \
[ENVO:01001240]\n- **env_medium**\n  Report the environmental material(s) immediately \
surrounding the sample or specimen at the time of sampling. We recommend using \
subclasses of 'environmental material' (http://purl.obolibrary.org/obo/ENVO_00010483). \
EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms from \
other OBO ontologies are permissible as long as they reference mass/volume nouns (e.g. \
air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a table \
top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  Allowed values: bark [PO:0004518], bulb \
[PO:0025356], corm [PO:0025355], ear infructescence axis [PO:0025623], flag leaf \
[PO:0020103], flower [PO:0009046], fruit [PO:0009001], leaf [PO:0025034], petiole \
[PO:0020038], phyllome [PO:0006001], pith [PO:0006109], plant callus [PO:0005052], \
plant gall [PO:0025626], plant litter [ENVO:01000628], pollen [PO:0025281], radicle \
[PO:0020031], rhizoid [PO:0030078], rhizome [PO:0004542], rhizosphere [ENVO:00005801], \
root [PO:0009005], root nodule [PO:0003023], sapwood [PO:0004513], secondary xylem \
[PO:0005848], seed [PO:0009010], seedling [PO:0008037], stem [PO:0009047], tuber \
[PO:0025522], xylem vessel [PO:0025417]\n\n---\n\n# SedimentInterface - Selected \
Fields\n- **env_broad_scale**\n  In this field, report which major environmental system \
your sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n  Allowed values: \
estuarine biome [ENVO:01000020], freshwater biome [ENVO:00000873], freshwater lake \
biome [ENVO:01000252], freshwater river biome [ENVO:01000253], large river delta biome \
[ENVO:00000889], mangrove biome [ENVO:01000181], marginal sea biome [ENVO:01000046], \
marine benthic biome [ENVO:01000024], marine biome [ENVO:00000447], marine cold seep \
biome [ENVO:01000127], marine coral reef biome [ENVO:01000049], marine neritic benthic \
zone biome [ENVO:01000025], marine salt marsh biome [ENVO:01000022], marine subtidal \
rocky reef biome [ENVO:01000050], xeric basin biome [ENVO:00000893]\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n  Allowed values: \
archipelago [ENVO:00000220], bank [ENVO:00000141], bar [ENVO:00000167], bay \
[ENVO:00000032], beach [ENVO:00000091], brackish estuary [ENVO:00002137], brackish lake \
[ENVO:00000540], cave [ENVO:00000067], coast [ENVO:01000687], coastal water body \
[ENVO:02000049], cold seep [ENVO:01000263], continental margin [ENVO:01000298], \
continental shelf [ENVO:00000223], cryoconite hole [ENVO:03000039], eutrophic lake \
[ENVO:01000548], fjord [ENVO:00000039], flood plain [ENVO:00000255], fumarole \
[ENVO:00000216], geyser [ENVO:00000050], hadalpelagic zone [ENVO:00000214], harbour \
[ENVO:00000463], hot spring [ENVO:00000051], hydrothermal seep [ENVO:01000265], \
hydrothermal vent [ENVO:00000215], hypersaline lake [ENVO:01001020], intertidal zone \
[ENVO:00000316], irrigation canal [ENVO:00000036], lake bed [ENVO:00000268], lentic \
water body [ENVO:01000617], littoral zone [ENVO:01000407], marine anoxic zone \
[ENVO:01000066], marine hydrothermal vent [ENVO:01000122], marine neritic zone \
[ENVO:00000206], marine sub-littoral zone [ENVO:01000126], mid-ocean ridge \
[ENVO:00000406], mud volcano [ENVO:00000402], ocean floor [ENVO:00000426], oil \
reservoir [ENVO:00002185], oil spill [ENVO:00002061], pond [ENVO:00000033], river \
[ENVO:00000022], river bank [ENVO:00000143], river bed [ENVO:00000384], saline \
evaporation pond [ENVO:00000055], saline lake [ENVO:00000019], saline pan \
[ENVO:00000279], sea floor [ENVO:00000482], sea grass bed [ENVO:01000059], shore \
[ENVO:00000304], spring [ENVO:00000027], stream [ENVO:00000023], stream bed \
[ENVO:00000383], submerged bed [ENVO:00000501]\n- **env_medium**\n  Report the \
environmental material(s) immediately surrounding the sample or specimen at the time of \
sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n  Allowed values: anaerobic sediment [ENVO:00002045], \
chemically contaminated sediment [ENVO:03600001], estuarine mud [ENVO:00002160], \
granular sediment [ENVO:01000117], hyperthermophilic sediment [ENVO:01000133], \
petroleum enriched sediment [ENVO:00002115], radioactive sediment [ENVO:00002154], \
sediment [ENVO:00002007], sediment permeated by saline water [ENVO:01001036], sludge \
[ENVO:00002044], thermophilic sediment [ENVO:01000132]\n\n---\n\n# SoilInterface - \
Selected Fields\n- **env_broad_scale**\n  In this field, report which major \
environmental system your sample or specimen came from. The systems identified should \
have a coarse spatial grain, to provide the general environmental context of where the \
sampling was done (e.g. were you in the desert or a rainforest?). We recommend using \
subclasses of ENVO's biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format \
(one term): termLabel [termID], Format (multiple terms): termLabel [termID]|termLabel \
[termID]|termLabel [termID]. Example: Annotating a water sample from the photic zone in \
middle of the Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. \
Example: Annotating a sample from the Amazon rainforest consider: tropical moist \
broadleaf forest biome [ENVO:01000228]. If needed, request new terms on the ENVO \
tracker, identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: \
string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n  Allowed \
values: alpine tundra biome [ENVO:01001505], anthropogenic terrestrial biome \
[ENVO:01000219], broadleaf forest biome [ENVO:01000197], coniferous forest biome \
[ENVO:01000196], cropland biome [ENVO:01000245], flooded grassland biome \
[ENVO:01000195], flooded savanna biome [ENVO:01000190], forest biome [ENVO:01000174], \
grassland biome [ENVO:01000177], mangrove biome [ENVO:01000181], mediterranean forest \
biome [ENVO:01000199], mediterranean grassland biome [ENVO:01000224], mediterranean \
savanna biome [ENVO:01000229], mediterranean shrubland biome [ENVO:01000217], \
mediterranean woodland biome [ENVO:01000208], mixed forest biome [ENVO:01000198], \
montane grassland biome [ENVO:01000194], montane savanna biome [ENVO:01000223], montane \
shrubland biome [ENVO:01000216], rangeland biome [ENVO:01000247], savanna biome \
[ENVO:01000178], shrubland biome [ENVO:01000176], subpolar coniferous forest biome \
[ENVO:01000250], subtropical broadleaf forest biome [ENVO:01000201], subtropical \
coniferous forest biome [ENVO:01000209], subtropical dry broadleaf forest biome \
[ENVO:01000225], subtropical grassland biome [ENVO:01000191], subtropical moist \
broadleaf forest biome [ENVO:01000226], subtropical savanna biome [ENVO:01000187], \
subtropical shrubland biome [ENVO:01000213], subtropical woodland biome \
[ENVO:01000222], temperate broadleaf forest biome [ENVO:01000202], temperate coniferous \
forest biome [ENVO:01000211], temperate grassland biome [ENVO:01000193], temperate \
mixed forest biome [ENVO:01000212], temperate savanna biome [ENVO:01000189], temperate \
shrubland biome [ENVO:01000215], temperate woodland biome [ENVO:01000221], terrestrial \
biome [ENVO:00000446], tidal mangrove shrubland [ENVO:01001369], tropical broadleaf \
forest biome [ENVO:01000200], tropical coniferous forest biome [ENVO:01000210], \
tropical dry broadleaf forest biome [ENVO:01000227], tropical grassland biome \
[ENVO:01000192], tropical mixed forest biome [ENVO:01001798], tropical moist broadleaf \
forest biome [ENVO:01000228], tropical savanna biome [ENVO:01000188], tropical \
shrubland biome [ENVO:01000214], tropical woodland biome [ENVO:01000220], tundra biome \
[ENVO:01000180], woodland biome [ENVO:01000175], xeric shrubland biome \
[ENVO:01000218]\n- **env_local_scale**\n  Report the entity or entities which are in \
the sample or specimen s local vicinity and which you believe have significant causal \
influences on your sample or specimen. We recommend using EnvO terms which are of \
smaller spatial grain than your entry for env_broad_scale. Terms, such as anatomical \
sites, from other OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are \
accepted in this field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  \
Allowed values: active permafrost layer [ENVO:04000009], agricultural field \
[ENVO:00000114], animal habitation [ENVO:00005803], anthropogenic litter \
[ENVO:03500005], aquifer [ENVO:00012408], area of cropland [ENVO:01000892], area of \
deciduous forest [ENVO:01000816], area of dwarf scrub [ENVO:01000861], area of \
evergreen forest [ENVO:01000843], area of pastureland or hayfields [ENVO:01000891], \
bank [ENVO:00000141], beach [ENVO:00000091], butte [ENVO:00000287], caldera \
[ENVO:00000096], canal [ENVO:00000014], cave [ENVO:00000067], channel [ENVO:03000117], \
cirque [ENVO:00000155], cliff [ENVO:00000087], crater [ENVO:00000514], delta \
[ENVO:00000101], desert [ENVO:01001357], dike [ENVO:01000671], ditch [ENVO:00000037], \
drainage basin [ENVO:00000291], dune [ENVO:00000170], estuary [ENVO:00000045], farm \
[ENVO:00000078], fen [ENVO:00000232], fjord [ENVO:00000039], flood plain \
[ENVO:00000255], frost heave [ENVO:01001568], fumarole [ENVO:00000216], garden \
[ENVO:00000011], glacier [ENVO:00000133], harbour [ENVO:00000463], hill \
[ENVO:00000083], hot spring [ENVO:00000051], hummock [ENVO:00000516], intertidal zone \
[ENVO:00000316], isthmus [ENVO:00000174], karst [ENVO:00000175], lake [ENVO:00000020], \
landfill [ENVO:00000533], levee [ENVO:00000178], mangrove swamp [ENVO:00000057], marsh \
[ENVO:00000035], mesa [ENVO:00000179], mine [ENVO:00000076], mountain [ENVO:00000081], \
mudflat [ENVO:00000192], needleleaf forest [ENVO:01000433], oil spill [ENVO:00002061], \
palsa [ENVO:00000489], park [ENVO:00000562], pasture [ENVO:00000266], peat swamp \
[ENVO:00000189], peatland [ENVO:00000044], peninsula [ENVO:00000305], plain \
[ENVO:00000086], plateau [ENVO:00000182], prairie [ENVO:00000260], quarry \
[ENVO:00000284], reservoir [ENVO:00000025], rhizosphere [ENVO:00005801], ridge \
[ENVO:00000283], river [ENVO:00000022], roadside [ENVO:01000447], shoreline \
[ENVO:00000486], sinkhole [ENVO:00000195], slope [ENVO:00002000], spring \
[ENVO:00000027], steppe [ENVO:00000262], stream [ENVO:00000023], tropical forest \
[ENVO:01001803], tunnel [ENVO:00000068], vadose zone [ENVO:00000328], volcano \
[ENVO:00000247], wadi [ENVO:00000031], watershed [ENVO:00000292], well [ENVO:00000026], \
wetland area [ENVO:00000043], woodland area [ENVO:00000109]\n- **env_medium**\n  Report \
the environmental material(s) immediately surrounding the sample or specimen at the \
time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  Allowed values: acidic soil [ENVO:01001185], \
acrisol [ENVO:00002234], agricultural soil [ENVO:00002259], albeluvisol \
[ENVO:00002233], alisol [ENVO:00002231], allotment garden soil [ENVO:00005744], \
alluvial paddy field soil [ENVO:00005759], alluvial soil [ENVO:00002871], alluvial \
swamp soil [ENVO:00005758], alpine soil [ENVO:00005741], andosol [ENVO:00002232], \
anthrosol [ENVO:00002230], arable soil [ENVO:00005742], arenosol [ENVO:00002229], bare \
soil [ENVO:01001616], beech forest soil [ENVO:00005770], bluegrass field soil \
[ENVO:00005789], bulk soil [ENVO:00005802], burned soil [ENVO:00005760], calcisol \
[ENVO:00002239], cambisol [ENVO:00002235], chernozem [ENVO:00002237], clay soil \
[ENVO:00002262], compacted soil [ENVO:06105205], compost soil [ENVO:00005747], cryosol \
[ENVO:00002236], dry soil [ENVO:00005748], durisol [ENVO:00002238], eucalyptus forest \
soil [ENVO:00005787], ferralsol [ENVO:00002246], fertilized soil [ENVO:00005754], \
fluvisol [ENVO:00002273], forest soil [ENVO:00002261], friable-frozen soil \
[ENVO:01001528], frost-susceptible soil [ENVO:01001638], frozen compost soil \
[ENVO:00005765], frozen soil [ENVO:01001526], gleysol [ENVO:00002244], grassland soil \
[ENVO:00005750], gypsisol [ENVO:00002245], hard-frozen soil [ENVO:01001525], heat \
stressed soil [ENVO:00005781], histosol [ENVO:00002243], jungle soil [ENVO:00005751], \
kastanozem [ENVO:00002240], lawn soil [ENVO:00005756], leafy wood soil [ENVO:00005783], \
leptosol [ENVO:00002241], limed soil [ENVO:00005766], lixisol [ENVO:00002242], loam \
[ENVO:00002258], luvisol [ENVO:00002248], manured soil [ENVO:00005767], meadow soil \
[ENVO:00005761], mountain forest soil [ENVO:00005769], muddy soil [ENVO:00005771], \
nitisol [ENVO:00002247], orchid soil [ENVO:00005768], ornithogenic soil \
[ENVO:00005782], paddy field soil [ENVO:00005740], pathogen-suppressive soil \
[ENVO:03600036], phaeozem [ENVO:00002249], planosol [ENVO:00002251], plastic-frozen \
soil [ENVO:01001527], plinthosol [ENVO:00002250], podzol [ENVO:00002257], pond soil \
[ENVO:00005764], red soil [ENVO:00005790], regosol [ENVO:00002256], rubber plantation \
soil [ENVO:00005788], savanna soil [ENVO:00005746], sawah soil [ENVO:00005752], soil \
[ENVO:00001998], solonchak [ENVO:00002252], solonetz [ENVO:00002255], spruce forest \
soil [ENVO:00005784], stagnosol [ENVO:00002274], surface soil [ENVO:02000059], \
technosol [ENVO:00002275], tropical soil [ENVO:00005778], ultisol [ENVO:01001397], \
umbrisol [ENVO:00002253], upland soil [ENVO:00005786], vegetable garden soil \
[ENVO:00005779], vertisol [ENVO:00002254]\n\n---\n\n# WastewaterSludgeInterface - \
Selected Fields\n- **env_broad_scale**\n  In this field, report which major \
environmental system your sample or specimen came from. The systems identified should \
have a coarse spatial grain, to provide the general environmental context of where the \
sampling was done (e.g. were you in the desert or a rainforest?). We recommend using \
subclasses of ENVO's biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format \
(one term): termLabel [termID], Format (multiple terms): termLabel [termID]|termLabel \
[termID]|termLabel [termID]. Example: Annotating a water sample from the photic zone in \
middle of the Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. \
Example: Annotating a sample from the Amazon rainforest consider: tropical moist \
broadleaf forest biome [ENVO:01000228]. If needed, request new terms on the ENVO \
tracker, identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: \
string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n- **env_medium**\n \
 Report the environmental material(s) immediately surrounding the sample or specimen at \
the time of sampling. We recommend using subclasses of 'environmental material' \
(http://purl.obolibrary.org/obo/ENVO_00010483). EnvO documentation about how to use the \
field: https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms \
from other OBO ontologies are permissible as long as they reference mass/volume nouns \
(e.g. air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a \
table top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[ENVO:\\d{7,8}\\]$`)\n\n---\n\n# WaterInterface - Selected Fields\n- \
**env_broad_scale**\n  In this field, report which major environmental system your \
sample or specimen came from. The systems identified should have a coarse spatial \
grain, to provide the general environmental context of where the sampling was done \
(e.g. were you in the desert or a rainforest?). We recommend using subclasses of ENVO's \
biome class: http://purl.obolibrary.org/obo/ENVO_00000428. Format (one term): termLabel \
[termID], Format (multiple terms): termLabel [termID]|termLabel [termID]|termLabel \
[termID]. Example: Annotating a water sample from the photic zone in middle of the \
Atlantic Ocean, consider: oceanic epipelagic zone biome [ENVO:01000033]. Example: \
Annotating a sample from the Amazon rainforest consider: tropical moist broadleaf \
forest biome [ENVO:01000228]. If needed, request new terms on the ENVO tracker, \
identified here: http://www.obofoundry.org/ontology/envo.html\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[ENVO:\\d{7,8}\\]$`)\n  Allowed values: \
aquatic biome [ENVO:00002030], concentration basin mediterranean sea biome \
[ENVO:01000004], dilution basin mediterranean sea biome [ENVO:01000128], epeiric sea \
biome [ENVO:01000045], estuarine biome [ENVO:01000020], freshwater biome \
[ENVO:00000873], freshwater lake biome [ENVO:01000252], freshwater river biome \
[ENVO:01000253], freshwater stream biome [ENVO:03605008], large freshwater lake biome \
[ENVO:00000891], large river biome [ENVO:00000887], large river delta biome \
[ENVO:00000889], large river headwater biome [ENVO:00000888], marginal sea biome \
[ENVO:01000046], marine abyssal zone biome [ENVO:01000027], marine basaltic \
hydrothermal vent biome [ENVO:01000054], marine bathyal zone biome [ENVO:01000026], \
marine benthic biome [ENVO:01000024], marine biome [ENVO:00000447], marine black smoker \
biome [ENVO:01000051], marine cold seep biome [ENVO:01000127], marine coral reef biome \
[ENVO:01000049], marine hadal zone biome [ENVO:01000028], marine hydrothermal vent \
biome [ENVO:01000030], marine neritic benthic zone biome [ENVO:01000025], marine \
pelagic biome [ENVO:01000023], marine reef biome [ENVO:01000029], marine salt marsh \
biome [ENVO:01000022], marine sponge reef biome [ENVO:01000123], marine subtidal rocky \
reef biome [ENVO:01000050], marine ultramafic hydrothermal vent biome [ENVO:01000053], \
marine upwelling biome [ENVO:01000858], marine white smoker biome [ENVO:01000052], \
mediterranean sea biome [ENVO:01000047], neritic epipelagic zone biome [ENVO:01000042], \
neritic mesopelagic zone biome [ENVO:01000043], neritic pelagic zone biome \
[ENVO:01000032], neritic sea surface microlayer biome [ENVO:01000041], ocean biome \
[ENVO:01000048], oceanic abyssopelagic zone biome [ENVO:01000038], oceanic bathypelagic \
zone biome [ENVO:01000037], oceanic benthopelagic zone biome [ENVO:01000040], oceanic \
epipelagic zone biome [ENVO:01000035], oceanic hadal pelagic zone biome \
[ENVO:01000039], oceanic mesopelagic zone biome [ENVO:01000036], oceanic pelagic zone \
biome [ENVO:01000033], oceanic sea surface microlayer biome [ENVO:01000034], small \
freshwater lake biome [ENVO:00000892], small river biome [ENVO:00000890], temperate \
marginal sea biome [ENVO:01000856], temperate marine upwelling biome [ENVO:01000860], \
temperate mediterranean sea biome [ENVO:01000857], tropical marginal sea biome \
[ENVO:01001230], tropical marine coral reef biome [ENVO:01000854], tropical marine \
upwelling biome [ENVO:01000859], xeric basin biome [ENVO:00000893]\n- \
**env_local_scale**\n  Report the entity or entities which are in the sample or \
specimen s local vicinity and which you believe have significant causal influences on \
your sample or specimen. We recommend using EnvO terms which are of smaller spatial \
grain than your entry for env_broad_scale. Terms, such as anatomical sites, from other \
OBO Library ontologies which interoperate with EnvO (e.g. UBERON) are accepted in this \
field. EnvO documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS\n  (type: string, \
pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  \
Allowed values: abyssal plain [ENVO:00000244], acid mine drainage [ENVO:00001997], \
agricultural field [ENVO:00000114], anoxic lake [ENVO:01001072], aquaculture farm \
[ENVO:03600074], aquifer [ENVO:00012408], archipelago [ENVO:00000220], beach \
[ENVO:00000091], biofilm [ENVO:00002034], black smoker [ENVO:00000218], cave \
[ENVO:00000067], coast [ENVO:01000687], cold seep [ENVO:01000263], continental margin \
[ENVO:01000298], coral reef [ENVO:00000150], cyanobacterial bloom [ENVO:03600071], \
desert spring [ENVO:02000139], epilimnion [ENVO:00002131], estuary [ENVO:00000045], fen \
[ENVO:00000232], fjord [ENVO:00000039], flood plain [ENVO:00000255], freshwater lake \
[ENVO:00000021], freshwater littoral zone [ENVO:01000409], freshwater river \
[ENVO:01000297], freshwater stream [ENVO:03605007], glacial lake [ENVO:00000488], \
glacier [ENVO:00000133], hadalpelagic zone [ENVO:00000214], harbour [ENVO:00000463], \
headwater [ENVO:00000153], hot spring [ENVO:00000051], hydrothermal vent \
[ENVO:00000215], hypolimnion [ENVO:00002130], inlet [ENVO:00000475], intertidal zone \
[ENVO:00000316], lake [ENVO:00000020], littoral zone [ENVO:01000407], mangrove swamp \
[ENVO:00000057], marine aphotic zone [ENVO:00000210], marine bathypelagic zone \
[ENVO:00000211], marine lake [ENVO:03600041], marine mesopelagic zone [ENVO:00000213], \
marine neritic zone [ENVO:00000206], marine pelagic zone [ENVO:00000208], marine photic \
zone [ENVO:00000209], marsh [ENVO:00000035], melt pond [ENVO:03000040], metalimnion \
[ENVO:00002132], microbial mat [ENVO:01000008], mine [ENVO:00000076], mine drainage \
[ENVO:00001996], mud volcano [ENVO:00000402], ocean [ENVO:00000015], ocean trench \
[ENVO:00000275], oceanic crust [ENVO:01000749], oil seep [ENVO:00002063], oil spill \
[ENVO:00002061], peatland [ENVO:00000044], pit [ENVO:01001871], pond [ENVO:00000033], \
puddle of water [ENVO:01000871], reservoir [ENVO:00000025], riffle [ENVO:00000148], \
river [ENVO:00000022], saline evaporation pond [ENVO:00000055], saline marsh \
[ENVO:00000054], sea [ENVO:00000016], shrimp pond [ENVO:01000905], sinkhole \
[ENVO:00000195], spring [ENVO:00000027], step pool [ENVO:03600096], strait \
[ENVO:00000394], stream [ENVO:00000023], stream pool [ENVO:03600094], stream run \
[ENVO:03600095], subglacial lake [ENVO:03000120], subterranean lake [ENVO:02000145], \
swamp ecosystem [ENVO:00000233], volcano [ENVO:00000247], water surface \
[ENVO:01001191], water tap [ENVO:03600052], water well [ENVO:01000002], wetland \
ecosystem [ENVO:01001209], whale fall [ENVO:01000140], wood fall [ENVO:01000142]\n- \
**env_medium**\n  Report the environmental material(s) immediately surrounding the \
sample or specimen at the time of sampling. We recommend using subclasses of \
'environmental material' (http://purl.obolibrary.org/obo/ENVO_00010483). EnvO \
documentation about how to use the field: \
https://github.com/EnvironmentOntology/envo/wiki/Using-ENVO-with-MIxS . Terms from \
other OBO ontologies are permissible as long as they reference mass/volume nouns (e.g. \
air, water, blood) and not discrete, countable entities (e.g. a tree, a leaf, a table \
top)\n  (type: string, pattern: `^([^\\s-]{1,2}|[^\\s-]+.+[^\\s-]+) \
\\[(ENVO:\\d{7,8}|PO:\\d{7})\\]$`)\n  Allowed values: acidic water [ENVO:01000358], \
alkaline water [ENVO:01000357], anoxic water [ENVO:01000173], bacon curing brine \
[ENVO:00003045], ballast water [ENVO:01000872], blue ice [ENVO:03000007], borax \
leachate [ENVO:00002142], bore hole water [ENVO:00003097], brackish water \
[ENVO:00002019], brine [ENVO:00003044], brown sea ice [ENVO:01001190], cloud water \
[ENVO:03600081], coastal sea water [ENVO:00002150], congelation ice in a fresh water \
body [ENVO:01001514], congelation sea ice [ENVO:01001512], contaminated water \
[ENVO:00002186], cooling water [ENVO:03600002], desalinated water [ENVO:06105269], \
distilled water [ENVO:00003065], ditch water [ENVO:00002158], drilling bore water \
[ENVO:00002159], drinking water [ENVO:00003064], epilithon [ENVO:03605001], epipelon \
[ENVO:03605002], epiphyton [ENVO:03605003], epipsammon [ENVO:03605004], epixylon \
[ENVO:03605005], erosionally enriched glacial ice [ENVO:03000005], erosionally enriched \
ice [ENVO:03000025], estuarine water [ENVO:01000301], eutrophic water [ENVO:00002224], \
first year ice [ENVO:03000071], fissure water [ENVO:01000940], frazil [ENVO:01001523], \
frazil ice [ENVO:03000046], fresh water [ENVO:00002011], freshwater congelation ice \
[ENVO:01001515], freshwater ice [ENVO:01001511], glacial ice [ENVO:03000004], \
groundwater [ENVO:01001004], hair ice [ENVO:01000847], highly saline water \
[ENVO:01001039], hydrothermal fluid [ENVO:01000134], hypereutrophic water \
[ENVO:01001018], hypersaline water [ENVO:00002012], hypoxic water [ENVO:01001064], ice \
cave congelation ice [ENVO:01001516], industrial wastewater [ENVO:01000964], \
interstitial water [ENVO:03600009], lake water [ENVO:04000007], leachate \
[ENVO:00002141], liquid water [ENVO:00002006], marine lake water [ENVO:03600042], \
marine snow [ENVO:01000158], meltwater [ENVO:01000722], mesotrophic water \
[ENVO:00002225], moderately saline water [ENVO:01001038], muddy water [ENVO:00005793], \
multiyear ice [ENVO:03000073], new ice [ENVO:03000063], oil field production water \
[ENVO:00002194], oligotrophic water [ENVO:00002223], oxic water [ENVO:01001063], \
permafrost congelation ice [ENVO:01001513], pond water [ENVO:00002228], powdery snow \
[ENVO:03000027], pulp-bleaching waste water [ENVO:00002193], rainwater [ENVO:01000600], \
residual water in soil [ENVO:06105238], river water [ENVO:01000599], runoff \
[ENVO:06105211], rural stormwater [ENVO:01001270], saline shrimp pond water \
[ENVO:01001257], saline water [ENVO:00002010], sea ice [ENVO:00002200], sea water \
[ENVO:00002149], second year ice [ENVO:03000072], sewage [ENVO:00002018], shuga \
[ENVO:03000075], slab snow [ENVO:03000108], slightly saline water [ENVO:01001037], snow \
[ENVO:01000406], spring water [ENVO:03600065], stagnant water [ENVO:03501370], sterile \
water [ENVO:00005791], stormwater [ENVO:01001267], stream water [ENVO:03605006], \
subterranean lake [ENVO:02000145], surface water [ENVO:00002042], tap water \
[ENVO:00003096], treated wastewater [ENVO:06105268], underground water [ENVO:00005792], \
urban stormwater [ENVO:01001268], waste water [ENVO:00002001], water ice \
[ENVO:01000277], water-body-derived ice [ENVO:01001557]
          """,
                "prompt": "{input}",
                "metrics": ["simple_question"],
            }
        },
        "matrix": {
            "hyperparameters": {
                "model": models,
                "temperature": [0.0],
            }
        },
        "cases": cases,
    }

    with open(args.output, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)

    print(f"Generated {len(cases)} cases x {len(models)} models -> {args.output}")


if __name__ == "__main__":
    main()
