# ADR 1: Block on several keys, then learn the match decision

## Context

Comparing every record with every other is quadratic: 7,500 records already make 28 million pairs.
A single blocking key (say, the same surname) misses every pair with a surname typo, a married
name, or first and last name swapped. Hand-written rules for the final decision get stuck between
two failures: too strict, and nicknames and typos break them; too loose, and fathers and sons with
the same name merge.

## Decision

- **Blocking on the union of seven keys:**
  - surname sound plus first initial, both ways round, so swapped names still meet;
  - the normalised email;
  - the phone number;
  - birth date plus surname initial;
  - birth year plus canonical first name;
  - postcode plus house number.
  Blocks larger than 150 records are skipped and counted, so one common key cannot explode the pair
  count.
- **A gradient-boosted classifier over 22 pair features.** It covers Jaro-Winkler on names with
  nicknames mapped to full names, Soundex, day/month-swapped and two-of-three birth dates, email
  local-part similarity, phone, street, house number and postcode. It also uses the *rarity* of a
  shared name: two records sharing "Smith" prove little, two sharing "Oyelaran" prove a lot.
- **Train and test on different people.** The matcher learns from half the population and is graded
  on the other half, so it cannot memorise anyone.

## Consequences

- Blocking keeps over 99% of true pairs while skipping over 99% of comparisons.
- The matcher's average precision on candidate pairs is above 0.99. Phone, birth date and email
  carry most of the signal, as the permutation importances in the report show.
- New fields (a national ID, a loyalty number) are new features and a retrain, not a new rule set.
