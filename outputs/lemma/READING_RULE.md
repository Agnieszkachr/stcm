# Lemma-based lexical baseline — reading rule (fixed before any lemma statistic was computed)

Recorded: 2026-10-23 15:29:39 UTC (2026-10-23 17:29 CEST)
Repository state at the time of writing: tag v1.3.0 = commit ec3f54d (no lemma code or lemma result existed).

## Question
Does the mean-centred cosine of Koine-DAPT still separate true Matthew–Luke parallels from
mismatched pairs once overlap of *lemmas* (not only of word forms) is removed?

## Data
MorphGNT SBLGNT (github.com/morphgnt/sblgnt, commit aaed91e57c8e4a8dc9a2383e129ca5e75fe6393d),
lemma column, verse ranges identical to the pericope table; composite units as the union of their
ranges; lemmas lower-cased and accent-stripped exactly as word forms are for the existing Jaccard.

## Primary test
Regress the centred-cosine pair matrix on the multiset Jaccard of lemmas over all 2,601
Matthew–Luke pairs of the double tradition; compare the residual of matched pairs with the
length-and-theme-matched permutation null (1,000 draws, seed 42 + 5, as for the word-form residual).

## Rule
- If the matched-pair residual exceeds the length-and-theme-matched null with p < 0.05, the claim
  of an embedding increment stands, qualified as "beyond lemma overlap".
- If not (p ≥ 0.05), the claim in the abstract, plain language summary, Sections 4.3, 5 and 6 is
  reformulated: the increment is largely a matter of inflection and vocabulary, and the paraphrase
  cases are reported descriptively.

Secondary analyses (reported whatever their outcome, not used to decide): random null; the same
test in the triple tradition; DT − TT on the residual; lemma TF-IDF; multiple regression on
Jaccard (forms) + Jaccard (lemmas) + TF-IDF (character 3–5-grams); ranks of the four paraphrase cases.
