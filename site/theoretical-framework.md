# Theoretical Framework

## The Synoptic Problem

The Synoptic Problem is the challenge of explaining the extensive literary similarities and differences among the first three canonical gospels: Matthew, Mark, and Luke. They share common wording, order of events, and even parenthetical editorial comments, which strongly implies a literary relationship rather than merely independent reliance on oral tradition.

## The Two-Source Hypothesis (2SH)

The most widely accepted solution in modern scholarship is the Two-Source Hypothesis, which posits:

1. **Markan Priority:** The Gospel of Mark was written first. Matthew and Luke independently used Mark as a primary narrative framework.
2. **The Q Source:** Matthew and Luke also share about 235 verses of sayings material (the "Double Tradition") not found in Mark (e.g., the Lord's Prayer, the Beatitudes). The 2SH posits they independently drew this material from a lost written document of Jesus's sayings, designated **Q** (from the German *Quelle*, meaning "source").

## Farrer Hypothesis & Other Alternatives

A significant minority of scholars reject Q, most notably through the **Farrer Hypothesis**. This model accepts Markan priority but argues that Luke used Matthew as a source (or vice versa), explaining the double tradition without postulating a lost document.

## How STCM Approaches the Problem

STCM does not attempt to "prove" Q. Instead, it tests whether the statistical distribution of the texts in high-dimensional embedding space is *consistent* with the Q hypothesis. 

If Matthew and Luke both used Mark, we can observe exactly what a "shared source" relationship looks like mathematically by analysing the Triple Tradition. We use this to calibrate two signatures:
- **Signature A:** The expected similarity between two texts drawn from a common written source.
- **Signature B:** The expected correlation in how two different authors modify their common source.

We then measure the Double Tradition in the same space. The comparison that matters is not the raw level of similarity but the **excess over each set's own floor**: how much more alike true parallels are than unrelated passages of the same kind already are. On that measure the double tradition exceeds the Markan calibration set, in both the corrected and the uncorrected geometry. This is consistent with derivation from a shared written source; it does not by itself decide between Q and Luke's direct use of Matthew.
