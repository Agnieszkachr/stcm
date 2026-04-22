"""
stcm/evaluation.py
==================
Evaluation and robustness module.

Performs:
1. Random permutation test — is the Q-score signal above chance?
2. Thematic-null permutation test — does Q-score exceed a *thematically
   matched* baseline, not merely a random one? (Addresses the circularity
   concern that topical overlap alone inflates similarity.)
3. Word-overlap baseline comparison — demonstrates that embedding-space
   analysis captures information beyond simple verbal-agreement percentages.
4. Sentence-level bootstrap robustness — perturbs input by resampling
   sentences within each pericope to measure Q-score stability under
   meaningful variation (replaces the uninformative seed-stability test).
5. Goulder redaction test — compares Q-score distributions of pericopes
   identified by Goulder (1989) as exhibiting Lukan redaction of Matthew
   against the remainder.
6. Internal BERT validation — checks that the embedding model produces
   sensible similarity judgements on known NT paraphrases and cross-gospel
   quotations.

All results written to outputs/reports/evaluation_summary.md
"""
from __future__ import annotations

import logging
import pathlib
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from stcm.calibration import CalibrationResult
from stcm.config import EvaluationConfig, default_config
from stcm.data_loader import Pericope, SynopticCorpus
from stcm.embeddings import EmbeddingPipeline
from stcm.scoring import QScorer, ScoringReport
from stcm.utils import cosine_similarity, normalise_greek, strip_accents

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thematic categories for the double tradition (§I.1 circularity fix)
# ---------------------------------------------------------------------------
# Each double-tradition pericope is assigned to a broad genre/thematic
# cluster so that the thematic-null can pair pericopes within the same
# category rather than at random.

THEMATIC_TAGS: Dict[str, str] = {
    "John's preaching":              "prophetic",
    "Temptation narrative (full)":   "narrative",
    "Beatitudes":                    "wisdom",
    "Love of enemies":               "wisdom",
    "Lord's Prayer":                 "liturgical",
    "Anxieties about life":          "wisdom",
    "Narrow gate":                   "wisdom",
    "Centurion's servant":           "narrative",
    "John's question from prison":   "prophetic",
    "Jesus on John":                 "prophetic",
    "Woes on Galilean cities":       "prophetic",
    "Hidden from wise revealed":     "wisdom",
    "Mission discourse":             "discipleship",
    "Harvest plentiful":             "discipleship",
    "Sign of Jonah":                 "prophetic",
    "Return of unclean spirit":      "prophetic",
    "Lamp of the body":              "wisdom",
    "Leaven of Pharisees":           "discipleship",
    "Fear of God not men":           "discipleship",
    "Blasphemy Holy Spirit":         "prophetic",
    "Thief in the night":            "apocalyptic",
    "Faithful servant":              "apocalyptic",
    "Not peace but sword":           "discipleship",
    "Reading the signs":             "apocalyptic",
    "Settling with opponent":        "wisdom",
    "Mustard seed and leaven":       "parable",
    "Many come from east west":      "apocalyptic",
    "Lament over Jerusalem":         "prophetic",
    "Parable of Great Banquet":      "parable",
    "Conditions of discipleship":    "discipleship",
    "Salt of the earth":             "wisdom",
    "Lost sheep":                    "parable",
    "Serving two masters":           "wisdom",
    "Day of the Son of Man":         "apocalyptic",
    "Talents / Minas":               "parable",
    "Judging twelve tribes":         "apocalyptic",
}

# ---------------------------------------------------------------------------
# Goulder redaction pericopes (§II.2)
# ---------------------------------------------------------------------------
# Pericopes that Goulder (1989) identifies as demonstrating clear Lukan
# redaction of Matthean material.  Under the Farrer Hypothesis these
# should show *directional* dependence signatures; under the Q hypothesis
# they should score comparably to non-Goulder pericopes.

GOULDER_REDACTION_LABELS: List[str] = [
    "Beatitudes",
    "Love of enemies",
    "Lord's Prayer",
    "Anxieties about life",
    "Narrow gate",
    "Faithful servant",
    "Not peace but sword",
    "Conditions of discipleship",
    "Parable of Great Banquet",
    "Talents / Minas",
]

# ---------------------------------------------------------------------------
# Known NT paraphrases for internal BERT validation (§III.6)
# ---------------------------------------------------------------------------
# Pairs of passages with known literary relationships (OT quotations in
# multiple gospels, Synoptic parallels, etc.) plus a control pair of
# unrelated passages.  Each entry: (label, gospel_a, ref_a, gospel_b, ref_b)
# where ref is (chapter, verse_start, verse_end).

NT_VALIDATION_PAIRS: List[Tuple[str, str, Tuple, str, Tuple]] = [
    # Synoptic parallels (known shared source — should be high similarity)
    ("Lord's Prayer: Matt vs Luke",     "matthew", (6, 9, 13),  "luke", (11, 2, 4)),
    ("Beatitudes: Matt vs Luke",        "matthew", (5, 3, 12),  "luke", (6, 20, 23)),
    ("Feeding 5000: Matt vs Mark",      "matthew", (14, 13, 21), "mark", (6, 30, 44)),
    ("Baptism: Mark vs Luke",           "mark",    (1, 9, 11),  "luke", (3, 21, 22)),
    # Control: unrelated passages (should show low similarity)
    ("Genealogy vs Crucifixion",        "matthew", (1, 1, 17),  "luke", (23, 26, 49)),
    ("Sermon Mount vs Passion",         "matthew", (5, 1, 12),  "mark", (15, 1, 15)),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PermutationTestResult:
    """Outcome of a permutation test."""
    observed_statistic: float
    null_distribution: np.ndarray = field(repr=False)
    p_value: float
    z_score: float
    description: str


@dataclass
class SensitivityResult:
    """Outcome of Q-score weight sensitivity analysis."""
    weight_configs: List[Tuple[float, float, float]]
    top5_per_config: List[List[str]]   # pericope labels for top-5
    mean_q_per_config: List[float]
    top5_stability: float              # Jaccard similarity of top-5 across configs
    description: str


@dataclass
class BootstrapRobustnessResult:
    """Outcome of sentence-level bootstrap robustness."""
    mean_q_scores: np.ndarray          # shape (n_pericopes,)
    bootstrap_std: np.ndarray          # shape (n_pericopes,)
    mean_bootstrap_std: float
    n_resamples: int
    description: str


@dataclass
class WordOverlapResult:
    """Comparison of embedding Q-scores vs. word-overlap agreement."""
    pericope_labels: List[str]
    embedding_q_scores: np.ndarray
    word_overlap_scores: np.ndarray
    correlation: float                  # Pearson r
    correlation_p: float
    residual_variance: float            # unexplained by word overlap
    description: str


@dataclass
class GoulderTestResult:
    """Comparison of Goulder-flagged vs. non-Goulder pericopes."""
    goulder_mean_q: float
    non_goulder_mean_q: float
    t_statistic: float
    p_value: float
    effect_size: float                  # Cohen's d
    goulder_labels: List[str]
    description: str


@dataclass
class BERTValidationResult:
    """Internal validation of embedding model on known NT paraphrases."""
    pair_labels: List[str]
    similarities: List[float]
    paraphrase_mean: float              # mean sim for known parallels
    control_mean: float                 # mean sim for unrelated pairs
    separation: float                   # paraphrase_mean - control_mean
    description: str


@dataclass
class EvaluationResult:
    """All evaluation results."""
    permutation_mean_q: PermutationTestResult
    permutation_top_q: PermutationTestResult
    thematic_null: Optional[PermutationTestResult]
    sensitivity: Optional[SensitivityResult]
    bootstrap_robustness: Optional[BootstrapRobustnessResult]
    word_overlap: Optional[WordOverlapResult]
    goulder_test: Optional[GoulderTestResult]
    bert_validation: Optional[BERTValidationResult]
    evaluation_summary_text: str


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Run robustness evaluation on the STCM pipeline.

    Parameters
    ----------
    pipeline    : EmbeddingPipeline
    calibration : CalibrationResult
    config      : EvaluationConfig
    """

    def __init__(
        self,
        calibration: CalibrationResult,
        pipeline: Optional[EmbeddingPipeline] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        self._cal = calibration
        self._pipe = pipeline or EmbeddingPipeline()
        self._cfg = config or default_config.evaluation

    # ------------------------------------------------------------------
    # Random permutation test (original)
    # ------------------------------------------------------------------

    def _permutation_test(
        self,
        report: ScoringReport,
        corpus: SynopticCorpus,
        stat_fn,
        description: str,
    ) -> PermutationTestResult:
        """
        Permute Matt–Luke pairings in the double tradition and recompute stat.

        The null hypothesis: Matt and Luke texts are matched randomly (no common
        source).  Under H0, the observed Q-score statistic should not be extreme.
        """
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]
        rng = np.random.default_rng(self._cfg.random_seed)

        observed = stat_fn(report.q_scores)
        null_stats: List[float] = []

        matt_texts = [p.matthew for p in double]
        luke_texts = [p.luke for p in double]

        log.info("Running %d permutations …", self._cfg.n_permutations)
        for i in range(self._cfg.n_permutations):
            perm_idx = rng.permutation(len(luke_texts))
            perm_qs: List[float] = []
            for j, m_txt in enumerate(matt_texts):
                l_txt = luke_texts[perm_idx[j]]
                e_m = self._pipe.embed_text(m_txt)
                e_l = self._pipe.embed_text(l_txt)
                cos = cosine_similarity(e_m, e_l)
                dev_a = cos - self._cal.sig_a.mean
                q = 0.5 * cos + 0.3 * max(0.0, dev_a)
                perm_qs.append(q)
            null_stats.append(stat_fn(np.array(perm_qs)))
            if (i + 1) % 100 == 0:
                log.info("  %d / %d permutations done", i + 1, self._cfg.n_permutations)

        null_arr = np.array(null_stats)
        p_value = float(np.mean(null_arr >= observed))
        z_score = float((observed - null_arr.mean()) / (null_arr.std() + 1e-12))
        log.info(
            "%s: observed=%.4f p=%.4f z=%.3f",
            description, observed, p_value, z_score,
        )
        return PermutationTestResult(
            observed_statistic=observed,
            null_distribution=null_arr,
            p_value=p_value,
            z_score=z_score,
            description=description,
        )

    # ------------------------------------------------------------------
    # Thematic-null permutation test (§I.1 — circularity fix)
    # ------------------------------------------------------------------

    def _thematic_null_test(
        self,
        report: ScoringReport,
        corpus: SynopticCorpus,
    ) -> PermutationTestResult:
        """
        A more demanding null model that pairs each Matthean pericope with
        a Lukan pericope from the *same thematic category* (e.g., wisdom
        saying with wisdom saying, apocalyptic with apocalyptic).

        This tests whether the correctly paired pericopes exceed what would
        be expected from mere topical similarity in Koine Greek, addressing
        the circularity concern raised in the referee report (§I.1).
        """
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]
        rng = np.random.default_rng(self._cfg.random_seed + 7)

        # Build thematic groups
        groups: Dict[str, List[int]] = {}
        for idx, p in enumerate(double):
            tag = THEMATIC_TAGS.get(p.label, "other")
            groups.setdefault(tag, []).append(idx)

        observed = float(np.mean(report.q_scores))
        null_stats: List[float] = []

        matt_texts = [p.matthew for p in double]
        luke_texts = [p.luke for p in double]

        n_perms = self._cfg.n_permutations
        log.info("Running %d thematic-null permutations …", n_perms)
        for i in range(n_perms):
            perm_qs: List[float] = []
            for j, m_txt in enumerate(matt_texts):
                # Find thematic group for this pericope
                tag = THEMATIC_TAGS.get(double[j].label, "other")
                group_indices = groups[tag]
                # Pick a random *different* Luke text from same thematic group
                candidates = [k for k in group_indices if k != j]
                if not candidates:
                    # Singleton group — fall back to any other pericope
                    candidates = [k for k in range(len(double)) if k != j]
                pick = rng.choice(candidates)
                l_txt = luke_texts[pick]
                e_m = self._pipe.embed_text(m_txt)
                e_l = self._pipe.embed_text(l_txt)
                cos = cosine_similarity(e_m, e_l)
                dev_a = cos - self._cal.sig_a.mean
                q = 0.5 * cos + 0.3 * max(0.0, dev_a)
                perm_qs.append(q)
            null_stats.append(float(np.mean(perm_qs)))
            if (i + 1) % 100 == 0:
                log.info("  %d / %d thematic permutations done", i + 1, n_perms)

        null_arr = np.array(null_stats)
        p_value = float(np.mean(null_arr >= observed))
        z_score = float((observed - null_arr.mean()) / (null_arr.std() + 1e-12))
        desc = "Thematic-null permutation test: mean Q-score"
        log.info(
            "%s: observed=%.4f thematic_null_mean=%.4f p=%.4f z=%.3f",
            desc, observed, null_arr.mean(), p_value, z_score,
        )
        return PermutationTestResult(
            observed_statistic=observed,
            null_distribution=null_arr,
            p_value=p_value,
            z_score=z_score,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Word-overlap baseline comparison (§I.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _word_overlap(text_a: str, text_b: str) -> float:
        """
        Compute word-level agreement between two Greek texts as the Jaccard
        coefficient of their word-token multisets.  Strips accents for
        approximate matching.
        """
        words_a = strip_accents(normalise_greek(text_a)).split()
        words_b = strip_accents(normalise_greek(text_b)).split()
        if not words_a or not words_b:
            return 0.0
        counter_a = Counter(words_a)
        counter_b = Counter(words_b)
        intersection = sum((counter_a & counter_b).values())
        union = sum((counter_a | counter_b).values())
        return intersection / union if union > 0 else 0.0

    def _word_overlap_comparison(
        self,
        report: ScoringReport,
        corpus: SynopticCorpus,
    ) -> WordOverlapResult:
        """
        Compare embedding-based Q-scores with simple word-level agreement
        percentages.  A low correlation would demonstrate that the embedding
        analysis captures information beyond lexical overlap (§I.2).
        """
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]
        labels: List[str] = []
        emb_qs: List[float] = []
        word_qs: List[float] = []

        score_map = {s.label: s.q_score for s in report.scores}
        for p in double:
            wo = self._word_overlap(p.matthew, p.luke)
            labels.append(p.label)
            emb_qs.append(score_map.get(p.label, 0.0))
            word_qs.append(wo)

        emb_arr = np.array(emb_qs)
        word_arr = np.array(word_qs)
        r, p_val = stats.pearsonr(emb_arr, word_arr)

        # Residual variance: how much of the embedding Q-score is NOT
        # explained by word overlap
        slope, intercept = np.polyfit(word_arr, emb_arr, 1)
        predicted = slope * word_arr + intercept
        residual_var = float(np.var(emb_arr - predicted))

        desc = "Word-overlap vs. embedding Q-score comparison"
        log.info(
            "%s: Pearson r=%.3f p=%.4f residual_var=%.4f",
            desc, r, p_val, residual_var,
        )
        return WordOverlapResult(
            pericope_labels=labels,
            embedding_q_scores=emb_arr,
            word_overlap_scores=word_arr,
            correlation=float(r),
            correlation_p=float(p_val),
            residual_variance=residual_var,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Sentence-level bootstrap robustness (§III.3 — replaces seed test)
    # ------------------------------------------------------------------

    def _sentence_bootstrap(
        self,
        report: ScoringReport,
        corpus: SynopticCorpus,
        n_resamples: int = 200,
    ) -> BootstrapRobustnessResult:
        """
        For each double-tradition pericope, split into sentences, resample
        sentences with replacement, re-embed, and recompute the Q-score.
        Reports per-pericope standard deviation across bootstrap iterations.

        This is a meaningful perturbation-based robustness check, unlike
        the seed-stability test which is trivially zero for a deterministic
        pipeline (§III.3).
        """
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]
        rng = np.random.default_rng(self._cfg.random_seed + 99)

        all_means: List[float] = []
        all_stds: List[float] = []

        log.info(
            "Running sentence-level bootstrap (%d resamples per pericope) …",
            n_resamples,
        )
        for p in double:
            # Split texts into sentence-like units (split on . ; · and Unicode periods)
            matt_sents = [s.strip() for s in re.split(r'[.;·]+', p.matthew) if s.strip()]
            luke_sents = [s.strip() for s in re.split(r'[.;·]+', p.luke) if s.strip()]
            if len(matt_sents) < 2 or len(luke_sents) < 2:
                # Too short to resample meaningfully — record observed score
                score_map = {s.label: s.q_score for s in report.scores}
                all_means.append(score_map.get(p.label, 0.0))
                all_stds.append(0.0)
                continue

            boot_qs: List[float] = []
            for _ in range(n_resamples):
                # Resample sentences with replacement
                m_idx = rng.choice(len(matt_sents), size=len(matt_sents), replace=True)
                l_idx = rng.choice(len(luke_sents), size=len(luke_sents), replace=True)
                m_text = " ".join(matt_sents[i] for i in m_idx)
                l_text = " ".join(luke_sents[i] for i in l_idx)
                e_m = self._pipe.embed_text(m_text)
                e_l = self._pipe.embed_text(l_text)
                cos = cosine_similarity(e_m, e_l)
                dev_a = cos - self._cal.sig_a.mean
                from stcm.utils import residual_vector
                r_m = residual_vector(e_m, self._cal.centroid_mark)
                r_l = residual_vector(e_l, self._cal.centroid_mark)
                resid_sim = cosine_similarity(r_m, r_l)
                q = 0.5 * cos + 0.3 * max(0.0, dev_a) + 0.2 * max(0.0, resid_sim)
                boot_qs.append(q)
            all_means.append(float(np.mean(boot_qs)))
            all_stds.append(float(np.std(boot_qs)))

        mean_arr = np.array(all_means)
        std_arr = np.array(all_stds)
        desc = "Sentence-level bootstrap robustness"
        log.info(
            "%s: mean_bootstrap_std=%.4f", desc, float(std_arr.mean()),
        )
        return BootstrapRobustnessResult(
            mean_q_scores=mean_arr,
            bootstrap_std=std_arr,
            mean_bootstrap_std=float(std_arr.mean()),
            n_resamples=n_resamples,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis (§III.1)
    # ------------------------------------------------------------------

    def _sensitivity_analysis(
        self,
        corpus: SynopticCorpus,
    ) -> SensitivityResult:
        """
        Recompute Q-scores under a grid of alternative weighting schemes
        and report whether the top-5 pericopes remain stable.
        """
        weight_configs: List[Tuple[float, float, float]] = [
            (0.5, 0.3, 0.2),   # default
            (0.6, 0.2, 0.2),   # more weight on raw cosine
            (0.4, 0.4, 0.2),   # more weight on deviation
            (0.4, 0.3, 0.3),   # more weight on residual
            (0.33, 0.33, 0.34), # equal weights
            (0.7, 0.15, 0.15), # cosine-dominated
            (0.5, 0.5, 0.0),   # no residual component
            (0.5, 0.0, 0.5),   # no deviation component
            (1.0, 0.0, 0.0),   # raw cosine only
        ]
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]

        top5_lists: List[List[str]] = []
        mean_qs: List[float] = []

        for w_cos, w_dev, w_res in weight_configs:
            scores: List[Tuple[str, float]] = []
            for p in double:
                e_m = self._pipe.embed_text(p.matthew)
                e_l = self._pipe.embed_text(p.luke)
                cos = cosine_similarity(e_m, e_l)
                dev_a = cos - self._cal.sig_a.mean
                from stcm.utils import residual_vector
                r_m = residual_vector(e_m, self._cal.centroid_mark)
                r_l = residual_vector(e_l, self._cal.centroid_mark)
                resid_sim = cosine_similarity(r_m, r_l)
                q = w_cos * cos + w_dev * max(0.0, dev_a) + w_res * max(0.0, resid_sim)
                scores.append((p.label, q))
            scores.sort(key=lambda x: x[1], reverse=True)
            top5 = [s[0] for s in scores[:5]]
            mean_q = float(np.mean([s[1] for s in scores]))
            top5_lists.append(top5)
            mean_qs.append(mean_q)
            log.debug(
                "  weights=(%.2f, %.2f, %.2f) mean_Q=%.4f top5=%s",
                w_cos, w_dev, w_res, mean_q, top5,
            )

        # Compute top-5 stability as mean pairwise Jaccard similarity
        n_configs = len(weight_configs)
        jaccards: List[float] = []
        for i in range(n_configs):
            for j in range(i + 1, n_configs):
                set_i = set(top5_lists[i])
                set_j = set(top5_lists[j])
                jacc = len(set_i & set_j) / len(set_i | set_j) if set_i | set_j else 0.0
                jaccards.append(jacc)
        stability = float(np.mean(jaccards)) if jaccards else 0.0

        desc = "Q-score weight sensitivity analysis"
        log.info("%s: top-5 Jaccard stability = %.3f", desc, stability)
        return SensitivityResult(
            weight_configs=weight_configs,
            top5_per_config=top5_lists,
            mean_q_per_config=mean_qs,
            top5_stability=stability,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Goulder redaction test (§II.2)
    # ------------------------------------------------------------------

    def _goulder_test(
        self,
        report: ScoringReport,
    ) -> GoulderTestResult:
        """
        Compare Q-score distributions of pericopes Goulder (1989) identifies
        as demonstrating Lukan redaction of Matthew against the rest.

        Under the Farrer Hypothesis, Goulder-flagged pericopes should show
        different (potentially lower) Q-scores if the method captures
        directional dependence.  Under the Q hypothesis, no systematic
        difference is predicted.
        """
        score_map = {s.label: s.q_score for s in report.scores}
        goulder_qs: List[float] = []
        non_goulder_qs: List[float] = []
        found_labels: List[str] = []

        for s in report.scores:
            if s.label in GOULDER_REDACTION_LABELS:
                goulder_qs.append(s.q_score)
                found_labels.append(s.label)
            else:
                non_goulder_qs.append(s.q_score)

        g_arr = np.array(goulder_qs) if goulder_qs else np.array([0.0])
        ng_arr = np.array(non_goulder_qs) if non_goulder_qs else np.array([0.0])

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(g_arr, ng_arr, equal_var=False)
        # Cohen's d
        pooled_std = np.sqrt(
            (g_arr.var() * len(g_arr) + ng_arr.var() * len(ng_arr))
            / (len(g_arr) + len(ng_arr))
        )
        d = float((g_arr.mean() - ng_arr.mean()) / (pooled_std + 1e-12))

        desc = "Goulder redaction test"
        log.info(
            "%s: goulder_mean=%.4f (n=%d) non_goulder_mean=%.4f (n=%d) "
            "t=%.3f p=%.4f d=%.3f",
            desc, g_arr.mean(), len(g_arr), ng_arr.mean(), len(ng_arr),
            t_stat, p_val, d,
        )
        return GoulderTestResult(
            goulder_mean_q=float(g_arr.mean()),
            non_goulder_mean_q=float(ng_arr.mean()),
            t_statistic=float(t_stat),
            p_value=float(p_val),
            effect_size=d,
            goulder_labels=found_labels,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Internal BERT validation (§III.6)
    # ------------------------------------------------------------------

    def _bert_validation(
        self,
        corpus: SynopticCorpus,
    ) -> BERTValidationResult:
        """
        Check whether Ancient-Greek-BERT produces sensible similarity
        judgements on known NT paraphrases and cross-gospel quotations.

        Known parallels should show high similarity; unrelated passages
        should show low similarity.  The gap between these two groups
        validates the model's semantic calibration for NT Greek.
        """
        from stcm.data_loader import SBLGNTLoader
        loader = SBLGNTLoader()
        books = loader._load_books()

        labels: List[str] = []
        sims: List[float] = []

        for label, gospel_a, ref_a, gospel_b, ref_b in NT_VALIDATION_PAIRS:
            from stcm.utils import verses_to_pericope
            text_a = verses_to_pericope(
                books[gospel_a], ref_a[0], ref_a[1], ref_a[2]
            )
            text_b = verses_to_pericope(
                books[gospel_b], ref_b[0], ref_b[1], ref_b[2]
            )
            if not text_a.strip() or not text_b.strip():
                log.warning("Empty text for validation pair: %s", label)
                continue
            e_a = self._pipe.embed_text(text_a)
            e_b = self._pipe.embed_text(text_b)
            sim = cosine_similarity(e_a, e_b)
            labels.append(label)
            sims.append(sim)
            log.debug("  BERT validation %-45s sim=%.3f", label, sim)

        # Split into paraphrase (first 4) and control (last 2) groups
        n_para = 4
        para_sims = sims[:n_para]
        ctrl_sims = sims[n_para:]
        para_mean = float(np.mean(para_sims)) if para_sims else 0.0
        ctrl_mean = float(np.mean(ctrl_sims)) if ctrl_sims else 0.0

        desc = "Internal BERT validation on known NT paraphrases"
        log.info(
            "%s: paraphrase_mean=%.3f control_mean=%.3f gap=%.3f",
            desc, para_mean, ctrl_mean, para_mean - ctrl_mean,
        )
        return BERTValidationResult(
            pair_labels=labels,
            similarities=sims,
            paraphrase_mean=para_mean,
            control_mean=ctrl_mean,
            separation=para_mean - ctrl_mean,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        report: ScoringReport,
        corpus: SynopticCorpus,
    ) -> EvaluationResult:
        """
        Run full evaluation suite.

        Parameters
        ----------
        report : ScoringReport from QScorer
        corpus : SynopticCorpus

        Returns
        -------
        EvaluationResult
        """
        # 1. Random permutation test
        log.info("Starting permutation test (mean Q-score) …")
        perm_mean = self._permutation_test(
            report, corpus, np.mean, "Permutation test: mean Q-score"
        )
        log.info("Starting permutation test (top-10 mean Q-score) …")
        perm_top = self._permutation_test(
            report, corpus,
            lambda qs: np.mean(np.sort(qs)[-10:]),
            "Permutation test: top-10 mean Q-score",
        )

        # 2. Thematic-null permutation test (§I.1)
        log.info("Starting thematic-null permutation test …")
        thematic = self._thematic_null_test(report, corpus)

        # 3. Sensitivity analysis (§III.1)
        log.info("Starting weight sensitivity analysis …")
        sensitivity = self._sensitivity_analysis(corpus)

        # 4. Sentence-level bootstrap (§III.3)
        log.info("Starting sentence-level bootstrap robustness …")
        bootstrap = self._sentence_bootstrap(report, corpus)

        # 5. Word-overlap comparison (§I.2)
        log.info("Starting word-overlap baseline comparison …")
        word_overlap = self._word_overlap_comparison(report, corpus)

        # 6. Goulder redaction test (§II.2)
        log.info("Starting Goulder redaction test …")
        goulder = self._goulder_test(report)

        # 7. Internal BERT validation (§III.6)
        log.info("Starting internal BERT validation …")
        bert_val = self._bert_validation(corpus)

        summary = self._build_summary(
            perm_mean, perm_top, thematic, sensitivity,
            bootstrap, word_overlap, goulder, bert_val,
        )
        return EvaluationResult(
            permutation_mean_q=perm_mean,
            permutation_top_q=perm_top,
            thematic_null=thematic,
            sensitivity=sensitivity,
            bootstrap_robustness=bootstrap,
            word_overlap=word_overlap,
            goulder_test=goulder,
            bert_validation=bert_val,
            evaluation_summary_text=summary,
        )

    def _build_summary(
        self,
        perm_mean: PermutationTestResult,
        perm_top: PermutationTestResult,
        thematic: PermutationTestResult,
        sensitivity: SensitivityResult,
        bootstrap: BootstrapRobustnessResult,
        word_overlap: WordOverlapResult,
        goulder: GoulderTestResult,
        bert_val: BERTValidationResult,
    ) -> str:
        # Sensitivity details
        sens_lines = []
        for i, (w, top5, mq) in enumerate(zip(
            sensitivity.weight_configs, sensitivity.top5_per_config,
            sensitivity.mean_q_per_config,
        )):
            tag = " (default)" if i == 0 else ""
            sens_lines.append(
                f"  ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}){tag}: "
                f"mean_Q={mq:.4f}  top-5: {', '.join(t[:25] for t in top5)}"
            )
        sens_block = "\n".join(sens_lines)

        # Bootstrap details
        boot_details = (
            f"mean_bootstrap_std={bootstrap.mean_bootstrap_std:.4f} "
            f"(across {bootstrap.n_resamples} resamples per pericope)"
        )

        # BERT validation
        bert_lines = []
        for lbl, sim in zip(bert_val.pair_labels, bert_val.similarities):
            bert_lines.append(f"  {lbl:<45s} sim={sim:.3f}")
        bert_block = "\n".join(bert_lines)

        return textwrap.dedent(f"""
# STCM Evaluation Summary

## 1. Permutation Test: Mean Q-Score (Random Null)

- Observed mean Q-score : {perm_mean.observed_statistic:.4f}
- Null mean             : {perm_mean.null_distribution.mean():.4f}
- Null std              : {perm_mean.null_distribution.std():.4f}
- p-value               : {perm_mean.p_value:.4f}
- z-score               : {perm_mean.z_score:.3f}
- Interpretation        : {"SIGNIFICANT (p < 0.05)" if perm_mean.p_value < 0.05 else "NOT SIGNIFICANT"}

## 2. Permutation Test: Top-10 Mean Q-Score

- Observed top-10 mean  : {perm_top.observed_statistic:.4f}
- p-value               : {perm_top.p_value:.4f}
- z-score               : {perm_top.z_score:.3f}
- Interpretation        : {"SIGNIFICANT (p < 0.05)" if perm_top.p_value < 0.05 else "NOT SIGNIFICANT"}

## 3. Thematic-Null Permutation Test (Circularity Check)

Pairs each Matthean pericope with a *thematically similar* Lukan pericope
(e.g., wisdom with wisdom, apocalyptic with apocalyptic) rather than a
random one.  This is a more demanding null that controls for topical
similarity in Koine Greek.

- Observed mean Q-score : {thematic.observed_statistic:.4f}
- Thematic null mean    : {thematic.null_distribution.mean():.4f}
- Thematic null std     : {thematic.null_distribution.std():.4f}
- p-value               : {thematic.p_value:.4f}
- z-score               : {thematic.z_score:.3f}
- Interpretation        : {"SIGNIFICANT — signal exceeds thematic baseline" if thematic.p_value < 0.05 else "NOT SIGNIFICANT against thematic null"}

## 4. Weight Sensitivity Analysis

Tests Q-score stability across nine alternative weighting schemes
(w_cosine, w_deviation, w_residual).  Top-5 Jaccard stability = {sensitivity.top5_stability:.3f}
(1.0 = identical top-5 across all schemes).

{sens_block}

## 5. Sentence-Level Bootstrap Robustness

Resamples sentences within each pericope (with replacement) and recomputes
Q-scores to measure stability under meaningful input perturbation.

- {boot_details}
- Interpretation        : {"ROBUST (mean std < 0.05)" if bootstrap.mean_bootstrap_std < 0.05 else "MODERATE SENSITIVITY to sentence composition"}

## 6. Word-Overlap vs. Embedding Q-Score Comparison

Compares embedding-based Q-scores with traditional word-level agreement
(Jaccard coefficient) to demonstrate that embeddings capture information
beyond simple verbal overlap.

- Pearson r             : {word_overlap.correlation:.3f}
- p-value               : {word_overlap.correlation_p:.4f}
- Residual variance     : {word_overlap.residual_variance:.4f}
- Interpretation        : {"Moderate correlation — embeddings partially track verbal agreement but capture additional semantic structure" if 0.3 < abs(word_overlap.correlation) < 0.8 else "Low correlation — embeddings capture substantially different information from word overlap" if abs(word_overlap.correlation) < 0.3 else "High correlation — embeddings largely track verbal agreement"}

## 7. Goulder Redaction Test

Compares Q-score distributions for pericopes Goulder (1989) identifies as
showing Lukan redaction of Matthew (n={len(goulder.goulder_labels)}) against the
remainder (n={len([s for s in range(36)]) - len(goulder.goulder_labels)}).

- Goulder mean Q-score  : {goulder.goulder_mean_q:.4f}
- Non-Goulder mean      : {goulder.non_goulder_mean_q:.4f}
- Welch's t             : {goulder.t_statistic:.3f}
- p-value               : {goulder.p_value:.4f}
- Cohen's d             : {goulder.effect_size:.3f}
- Interpretation        : {"Significant difference" if goulder.p_value < 0.05 else "No significant difference"} between Goulder-flagged and non-Goulder pericopes

## 8. Internal BERT Validation (Known NT Paraphrases)

Tests embedding-model calibration on known synoptic parallels (expected
high similarity) vs. unrelated passage pairs (expected low similarity).

{bert_block}

- Paraphrase mean       : {bert_val.paraphrase_mean:.3f}
- Control mean          : {bert_val.control_mean:.3f}
- Separation            : {bert_val.separation:.3f}
- Interpretation        : {"GOOD — model discriminates known parallels from unrelated text" if bert_val.separation > 0.1 else "WEAK — model may not adequately capture NT Greek semantics"}

## Caveats

- These results depend on the quality and coverage of the embedding model.
- Permutation tests assume exchangeability — valid for cosine-based
  statistics but may not hold if texts have structural dependencies.
- The thematic-null categories are broad; finer-grained genre tagging
  might yield a stricter baseline.
- A low p-value does NOT prove Q exists; it shows the embedding geometry
  is consistent with the Q hypothesis relative to both random and
  thematic baselines.
        """).strip()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def evaluate_and_save(
    report: ScoringReport,
    corpus: SynopticCorpus,
    calibration: CalibrationResult,
    pipeline: Optional[EmbeddingPipeline] = None,
    config: Optional[EvaluationConfig] = None,
    out_path: Optional[pathlib.Path] = None,
) -> EvaluationResult:
    """
    Run evaluation and save the summary report.

    Parameters
    ----------
    report      : ScoringReport
    corpus      : SynopticCorpus
    calibration : CalibrationResult
    pipeline    : EmbeddingPipeline (or None)
    config      : EvaluationConfig (or None)
    out_path    : path for .md report (default outputs/reports/evaluation_summary.md)

    Returns
    -------
    EvaluationResult
    """
    out_path = out_path or (
        default_config.paths.outputs_reports / "evaluation_summary.md"
    )
    engine = EvaluationEngine(calibration=calibration, pipeline=pipeline, config=config)
    result = engine.evaluate(report, corpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.evaluation_summary_text, encoding="utf-8")
    log.info("Evaluation summary saved to %s", out_path)
    return result
