# STCM — Synoptic Transform Calibration Model

Welcome to the documentation for the **Synoptic Transform Calibration Model**, a computational research framework for investigating the Synoptic Problem through NLP embedding analysis.

## What is this?

STCM tests whether a transformer representation of Koine Greek can measure literary dependence — first on relations that are known (Matthew and Luke copying Mark), then on the disputed *double tradition* (passages found in Matthew and Luke but absent from Mark), and asks what it adds to counting shared words.

## How does it work?

```mermaid
graph TD
    A[SBLGNT Greek Text] --> B[Koine-Greek-BERT Embeddings]
    B --> C[Triple-Tradition Calibration]
    C --> D[Two-step: Matt-Luke]
    C --> E[One-step: Luke-Mark, Matt-Mark]
    D --> F[Double tradition: excess over own floor]
    E --> F
    F --> H[Permutation nulls, CIs, Mark-Q overlaps, sayings control]
    F --> J[Lexical baselines: Jaccard on forms and lemmas, TF-IDF]
    F --> G[Covariates & directionality controls]
    J --> I
    G --> I[Results & Reports]
    H --> I
```

## Quick navigation

- **[Theoretical Framework](theoretical-framework.md)** — The Synoptic Problem and the Q hypothesis
- **[Methodology](methodology.md)** — Mathematical formulation and pipeline design
- **[Installation](installation.md)** — Setup instructions
- **[Usage](usage.md)** — Running the pipeline
- **[Results](results.md)** — Findings and interpretation
- **[FAQ](faq.md)** — Common questions
