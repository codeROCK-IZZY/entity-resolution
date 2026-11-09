# ADR 2: Cluster with average linkage, not connected components

## Context

Most entity-resolution pipelines turn pair decisions into entities by taking connected components:
if A matches B and B matches C, then A, B and C are one person. One wrong link is enough to chain
two different people together, and look-alikes provide wrong links constantly. A son shares his
father's name and address. A twin shares a surname, an address and a birthday. On this data,
connected components merge a third of twin pairs and nearly half of father-son pairs.

## Decision

Agglomerate greedily, strongest link first. Two clusters merge only if the *average* match
probability over every record pair between them clears the threshold. Pairs blocking never proposed
are scored on demand rather than assumed to match.

## Consequences

- One strong link can no longer pull in a whole household. The number of clusters mixing different
  people falls from 54 to 13, and twin merges fall from 33% to 1.3%.
- Fathers and sons with the same name remain the hardest case. Their merges fall from 45% to 21%,
  because the only reliable difference is the birth date, and it is missing from 20% of records.
  Telling them apart needs another field (a generation suffix, an account number).
- Recall drops slightly, because a person whose records are very inconsistent may stay split. In the
  threshold table, F1 is flat between 0.3 and 0.5, and 0.5 is used.
- Cost grows with cluster size, since cross-cluster pairs are scored on demand. Clusters here are a
  handful of records, so the whole test half resolves in seconds.
