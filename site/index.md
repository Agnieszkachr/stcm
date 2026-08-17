# STCM — Synoptic Transform Calibration Model

Welcome to the documentation for the **Synoptic Transform Calibration Model**, a computational research framework for investigating the Synoptic Problem through NLP embedding analysis.

## What is this?

STCM uses transformer-based embeddings of Ancient Greek text to test whether the *double-tradition* material (passages found in Matthew and Luke but absent from Mark) exhibits an embedding-space signature consistent with derivation from a shared written source — the hypothetical **Q document** (from German *Quelle*, "source").

## How does it work?

```mermaid
graph TD
    A[SBLGNT Greek Text] --> B[Koine-Greek-BERT Embeddings]
    B --> C[Triple-Tradition Calibration]
    C --> D[Signature A: Independent baseline]
    C --> E[Signature B: Residual dependence]
    D --> F[Double-Tradition Scoring: centred cosine]
    E --> F
    F --> G[Latent Q Reconstruction]
    F --> H[Permutation Testing]
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
