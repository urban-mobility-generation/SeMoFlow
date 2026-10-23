# SeMoFlow

SeMoFlow generates weekly mobility trajectories as timed visits to POIs in a
city catalog.

This repository contains:

- The full pipeline: [preprocessing](#data-and-preprocessing),
  [training](#training), and [sampling](#sampling), runnable end to end on
  the public Foursquare dataset.
- [Trained checkpoints](#released-checkpoints) for Boston, Chicago, Dallas,
  New York City, and Pittsburgh.
- [30,000 generated trajectories per city](#sample-trajectories) from those
  checkpoints.

## Method

### Motivation

People revisit places at different levels: the same POI, another POI in the
same category, or a place with a similar function. SeMoFlow represents POIs
with hierarchical Semantic IDs (SIDs) so a model can learn both shared
patterns and place-specific behavior.

![Motivation for hierarchical POI representations and repeat-visit modeling](assets/motivation.png)

### Framework

The planner samples visit times and a coarse SID prefix for each visit. A
conditional flow-matching realizer completes the SID sequence. A catalog
decoder maps the resulting IDs back to POIs.

![SeMoFlow framework: semantic tokenizer, autoregressive prefix planner, suffix latent-flow realizer, and POI grounding](assets/framework-v2.png)

The tokenizer learns POI representations from text and can include geographic
features and co-visitation graph embeddings. Each model stage is trained
separately.

### Semantic IDs

The residual quantizer builds each SID from successive codebooks. Early codes
form a coarse prefix shared by related places; later codes distinguish more
specific POIs.

![Residual quantization of POI embeddings into coarse prefixes and fine-grained SID suffixes](assets/semantic-ids.png)

## Install

The dependency environment targets Linux x86_64 with Python 3.12–3.14.
Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Reading the [sample trajectories](#sample-trajectories) needs only a Parquet
reader, not this environment.

## Data and preprocessing

Download the Foursquare global 2015 release from the
[dataset source page](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
and extract it under `data/foursquare_global_2015/`. The directory should
contain `dataset_TIST2015_Checkins.txt`, `dataset_TIST2015_POIs.txt`, and
`dataset_TIST2015_Cities.txt`. Then run:

```bash
uv run semoflow-preprocess --config configs/preprocess/foursquare.yaml
```

To read the release from another directory, add
`--set data.foursquare.archive_path=/path/to/extracted/release`.

The [preprocessing config](configs/preprocess/foursquare.yaml) selects
Los Angeles, Washington, D.C., Atlanta, San Francisco, and Milwaukee. The run
writes `processed_visits.parquet`, `quality_scores.parquet`, and
`places_index.parquet` to `outputs/preprocessing/foursquare_five_city/`.

## Training

Training runs in three stages, each a separate command, and continues from
the preprocessing output above. The sequence was checked end to end. Set
`--device` and the `device` fields in the YAML files for your hardware;
every command documents its options under `--help`.

| Stage | Trains | Main outputs |
| --- | --- | --- |
| [1. Tokenizer](#stage-1-poi-encoders-and-sid-tokenizer) | Text and graph POI encoders, then the SID tokenizer | `$T/placekey_lookup.parquet`, `$T/tokenizer_codebooks.pt`, `$T/grounding_index.pkl` |
| [2. Planner](#stage-2-planner) | Autoregressive prefix planner | `outputs/planner/best.pt` |
| [3. Realizer](#stage-3-realizer) | Suffix latent-flow realizer | `outputs/realizer/best.pt` |

All training and sampling commands use these two paths:

```bash
P=outputs/preprocessing/foursquare_five_city
T=outputs/tokenization/foursquare_five_city/dense_f_full
```

### Stage 1: POI encoders and SID tokenizer

Train the text and graph POI encoders:

```bash
uv run semoflow-train encoder --config configs/encoder/foursquare_five_city.yaml
```

Train the tokenizer, which writes its outputs to `$T`:

```bash
uv run semoflow-train tokenizer \
  --config configs/tokenization/multi_city/foursquare_five_city/dense_full_transfer.yaml
```

### Stage 2: Planner

Split visits into weekly segments:

```bash
uv run semoflow-artifacts build-weekly-segments \
  --stays $T/augmented_processed_visits.parquet --output outputs/segments.parquet
```

Build the planner dataset. Multi-city catalogs need city context
(`--context-mode city`) so samples can be assigned to a city:

```bash
uv run semoflow-artifacts build-occurrence-planner-dataset \
  --segments outputs/segments.parquet --lookup $T/placekey_lookup.parquet \
  --codebook-size 48 --context-mode city --output outputs/planner_dataset.pt
```

Train the planner:

```bash
uv run semoflow-train planner \
  --dataset outputs/planner_dataset.pt --output-dir outputs/planner
```

### Stage 3: Realizer

Build the realizer bank:

```bash
uv run semoflow-artifacts build-realizer-bank \
  --lookup $T/placekey_lookup.parquet --segments outputs/segments.parquet \
  --output outputs/bank.pt
```

Train the realizer:

```bash
uv run semoflow-train realizer --bank outputs/bank.pt \
  --codebooks $T/tokenizer_codebooks.pt --codebook-size 48 \
  --catalog-path $T/placekey_lookup.parquet --output-dir outputs/realizer
```

The other files in [`configs/`](configs/) record the encoder and tokenizer
recipes for the Veraset cities. They read processed Veraset data, which is
not distributed (see [Released checkpoints](#released-checkpoints)).

## Sampling

Sampling has two steps: the planner samples visit times and SID prefixes,
then the realizer completes the SIDs and grounds them to catalog POIs. Set
`N` to the number of trajectories to generate:

```bash
N=1000
```

### Step 1: Sample plans

Prepare `CITIES.parquet` with one row per sample and a `city_id` column
naming the city to sample for. City IDs are defined in
[`constants.py`](src/semoflow/config/constants.py):

| City | `city_id` |
| --- | --- |
| Los Angeles | 7 |
| Washington, D.C. | 8 |
| Atlanta | 9 |
| San Francisco | 10 |
| Milwaukee | 11 |

Turn it into planner conditions:

```bash
uv run semoflow-artifacts build-occurrence-planner-conditions \
  --checkpoint outputs/planner/best.pt --input CITIES.parquet \
  --output outputs/conditions.pt
```

Sample plans:

```bash
uv run semoflow-eval sample-plans --checkpoint outputs/planner/best.pt \
  --conditions outputs/conditions.pt --num-samples $N --output outputs/plans.pt
```

### Step 2: Compose POI trajectories

Complete the SIDs, ground them to POIs, and evaluate the result:

```bash
uv run semoflow-eval compose --realizer-checkpoint outputs/realizer/best.pt \
  --planner-samples outputs/plans.pt --planner-checkpoint outputs/planner/best.pt \
  --bank outputs/bank.pt --codebooks $T/tokenizer_codebooks.pt \
  --lookup $T/placekey_lookup.parquet --grounding-index $T/grounding_index.pkl \
  --places $P/places_index.parquet --max-samples $N --output outputs/composition.json
```

### Sampling from the released checkpoints

The released planner checkpoints sample on their own:

```bash
uv run semoflow-eval sample-plans \
  --checkpoint veraset_checkpoints/chicago/planner.pt \
  --output outputs/samples/chicago_plans.pt --num-samples 1000
```

Composing these plans into grounded POI trajectories also needs the city's
validation bank, SID lookup, and places table. Those are derived from the
licensed data and are not released, so the composed trajectories are
released instead; see [Sample trajectories](#sample-trajectories).

## Released checkpoints

[`veraset_checkpoints/`](veraset_checkpoints/) holds one directory of trained
models per city. They were trained on Veraset visit data. Its license does
not allow redistribution, so neither the raw nor the processed Veraset data
is included, and the public preprocessing pipeline covers Foursquare only.

| City | Planner | Realizer | Grounding index |
| --- | --- | --- | --- |
| Boston | [24 MB](veraset_checkpoints/boston/planner.pt) | [38 MB](veraset_checkpoints/boston/realizer.pt) | [15 MB](veraset_checkpoints/boston/grounding.pkl) |
| Chicago | [25 MB](veraset_checkpoints/chicago/planner.pt) | [38 MB](veraset_checkpoints/chicago/realizer.pt) | [25 MB](veraset_checkpoints/chicago/grounding.pkl) |
| Dallas | [53 MB](veraset_checkpoints/dallas/planner.pt) | [38 MB](veraset_checkpoints/dallas/realizer.pt) | [22 MB](veraset_checkpoints/dallas/grounding.pkl) |
| New York City | [25 MB](veraset_checkpoints/new_york_city/planner.pt) | [38 MB](veraset_checkpoints/new_york_city/realizer.pt) | [47 MB](veraset_checkpoints/new_york_city/grounding.pkl) |
| Pittsburgh | [24 MB](veraset_checkpoints/pittsburgh/planner.pt) | [38 MB](veraset_checkpoints/pittsburgh/realizer.pt) | [8 MB](veraset_checkpoints/pittsburgh/grounding.pkl) |

Each city directory contains:

| File | Contents |
| --- | --- |
| `planner.pt` | The autoregressive prefix planner. |
| `realizer.pt` | The suffix latent-flow realizer, with its tokenizer codebooks embedded. |
| `grounding.pkl` | The index that maps full SIDs to catalog POIs (placekey, category, NAICS code, and coordinates). |

## Sample trajectories

[`sampled_data/`](sampled_data/) holds 30,000 generated trajectories for each
city, one file per city named `generated_poi_trajectories_30k.parquet`. The
files contain no real device identifiers or timestamps.

| City | Trajectories | Visits | Size |
| --- | --- | --- | --- |
| [Boston](sampled_data/boston/generated_poi_trajectories_30k.parquet) | 30,000 | 700,376 | 13 MB |
| [Chicago](sampled_data/chicago/generated_poi_trajectories_30k.parquet) | 30,000 | 701,044 | 15 MB |
| [Dallas](sampled_data/dallas/generated_poi_trajectories_30k.parquet) | 30,000 | 724,826 | 15 MB |
| [New York City](sampled_data/new_york_city/generated_poi_trajectories_30k.parquet) | 30,000 | 753,268 | 18 MB |
| [Pittsburgh](sampled_data/pittsburgh/generated_poi_trajectories_30k.parquet) | 30,000 | 629,518 | 11 MB |

Each row is one visit:

| Column | Description |
| --- | --- |
| `trajectory_id` | The generated trajectory the visit belongs to. |
| `event_index` | The visit's position within its trajectory. |
| `placekey_id`, `placekey` | The grounded POI. |
| `top_category`, `naics_code` | The POI's category and NAICS code. |
| `latitude`, `longitude` | The POI's coordinates. |
| `sem_code_0`–`sem_code_5` | The visit's SID, coarse to fine. |
| `grounding_candidate_count` | How many catalog POIs match the SID. |
| `poi_selection` | How one POI was chosen when several match (sampled from a prior fitted on training visits). |

To read one city's file and order each trajectory's visits, for example
with [Polars](https://pola.rs/):

```python
import polars as pl

visits = pl.read_parquet(
    "sampled_data/chicago/generated_poi_trajectories_30k.parquet"
).sort("trajectory_id", "event_index")
```
