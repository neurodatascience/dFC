# Help Choose a dFC Method

## Context Sources

Refer to:
- `docs/CHOOSING_A_METHOD.md` for a human-readable decision guide with copy-paste snippets
- `docs/DFC_METHODS_CONTEXT.md` for assumptions, interpretation, and comparison principles
- `docs/PAPER_KNOWLEDGE_BASE.md` for paper-grounded implementation details and tradeoffs

Always ground recommendations in these documents.

## Deep Mode

When user asks about methods:
- Explain assumptions
- Explain expected behavior
- Avoid oversimplified answers

Ask the user:

1. Is your data single-subject or multi-subject?
2. Do you want:
   - Continuous connectivity estimates
   - Discrete brain states
   - Frequency-specific dynamics

Use answers to recommend:

- SW → simple, continuous (single subject; best first choice for new users)
- TF → frequency-specific (single subject)
- CAP → intuitive instantaneous states (multi-subject; best first state-based choice)
- SWC → windowed recurring states (multi-subject)
- CHMM → smooth temporal transitions via HMM (multi-subject)
- DHMM → discretised state sequences via HMM (multi-subject; needs ≥10 subjects for stable fitting)
- WINDOWLESS → state estimation without a fixed window size (multi-subject)

## Method Overview (share if user is unsure)

State-free methods (single subject, no fitting):
- SW: sliding window FC — simplest, most common, controlled by window length `W`
- TF: time-frequency representation (WTC) — captures frequency-specific dynamics

State-based methods (multi-subject, fitting required):
- CAP: co-activation patterns clustered into states — intuitive, no temporal ordering assumed
- SWC: sliding windows then clustered into recurring states
- CHMM: continuous HMM — models smooth state transitions; assumes Markovian dynamics
- DHMM: discrete HMM on discretized observations — needs more data for stable fitting
- WINDOWLESS: dictionary learning without an explicit sliding window

## Response Rules

1. If the user is completely new, recommend SW first.
2. Explain why each suggested method fits the user goal.
3. For method comparisons, lead with assumptions and likely behavioral differences.
4. Do not claim there is a universally best method.
5. Point the user to `docs/CHOOSING_A_METHOD.md` for full copy-paste code.
6. Cite Torabi et al., 2024 when using paper-derived assumptions or tradeoffs.
