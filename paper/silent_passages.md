# Silent passages are causal, not noise

A *silent* passage is causal (removing it flips the answer's value) but not salient (it does
not state the wrong value). Because the oracle compares normalized answer *values*, a flip
driven only by rephrasing is never counted, so the causal label already excludes surface jitter.

**Effect size.** Removing a silent passage moves the answer log-probability about as much as
removing a true culprit, and far above the near-zero shift a perturbation artifact would give:

- silent passages: n=2881, median=2.34, mean=5.66, share below 0.5 logprob (jitter range)=13%
- culprit passages: n=1482, median=6.11, mean=10.96, share below 0.5 logprob (jitter range)=6%

**Candidate silent drivers** — the passage bridges to a *different* entity than the wrong
answer (its title shares no word with the wrong answer) and the wrong value is absent under
dash/abbreviation-normalized matching. Still verify by eye: the silent label is sensitive to the
salience matcher, and many silents elsewhere are matcher near-misses where the passage states
the wrong value in a variant form (so report the silent share under the strict matcher too).

- *What is the place of birth of Remigius Of Rouen's father?*
  - gold `Herstal`; model answered `Chênex (for Charles Martel, Remigius' father)` (wrong). Silent passage `Joseph Duval` (does not state `Chênex (for Charles Martel, Remigius' father)`):
    > Joseph Marie Louis Duval (11 October 1928 – 23 May 2009) was the French Roman Catholic Archbishop of the Roman Catholic Archdiocese of Rouen. Born in Chênex, Duval was ordained to the priesthood on June 8, 1953. On May 14, 1974, Pope Paul V
  - removing it flips the answer to `Pepin of Herstal (Charles Martel)` (Delta logprob 40.7) -- it drives the wrong hop without stating the wrong value.
- *on 16 January 1995 "First Knight" was being filmed at a studio located how far from Windsor ?*
  - gold `7 miles`; model answered `early 1970s (from [2] and [6])` (wrong). Silent passage `1995 Benson and Hedges Open` (does not state `early 1970s (from [2] and [6])`):
    > The 1995 Benson and Hedges Open was a men's tennis tournament held in Auckland, New Zealand and played on outdoor hard courts.  The event was part of the World Series of the 1995 ATP Tour.  It was the 28th edition of the tournament and was
  - removing it flips the answer to `7 miles` (Delta logprob 29.2) -- it drives the wrong hop without stating the wrong value.
- *What is the place of birth of the performer of song Damage (Pharoahe Monch Song)?*
  - gold `Queens`; model answered `The Bronx, New York` (wrong). Silent passage `Billy Milano` (does not state `The Bronx, New York`):
    > Billy Milano is a Bronx- born heavy metal musician now based in Austin, Texas. He is the singer and- occasionally- guitarist and bassist of crossover thrash band M.O.D., and he was also the singer of its predecessor, Stormtroopers of Death.
  - removing it flips the answer to `South Jamaica, Queens, New York` (Delta logprob 27.0) -- it drives the wrong hop without stating the wrong value.
