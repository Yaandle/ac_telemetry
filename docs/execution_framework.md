# Execution Framework (Fibonacci Ladder)

This project follows a **Fibonacci Ladder execution framework** —
each stage changes *how* the work is approached, not just *what* is built.

The goal is completion, iteration, and system understanding over novelty.

---

## Stage 1 — Foundational Win ✅

**Goal:** Finish something small, real, and runnable.

- Single language (Python)
- Single toolchain
- Real input → real output
- Fully working and documented

**Outcome:**  
`reader.py` captures live ACC telemetry at **100 Hz** and logs structured CSV data.

---

## Stage 2 — Repetition + Variation ✅

**Goal:** Build the same thing twice with small, meaningful variation.
Stage 2 consists of two parallel analytics implementations, both built directly on top of the same telemetry reader output and shared analysis logic.

- Reuse ≥70% of code
- Change one dimension at a time
- Learn what matters vs. what doesn't

**Outcome:**            
`telemetry_analysis_streamlit.py` provides UI visualisations and analytics;         
`telemetry_analysis_terminal.py` provides CLI-based analytics and ML dataset preparation.

---

## Stage 3 — System Formation 🧩

**Goal:** Turn working scripts into a coherent, reusable system.

This repository intentionally includes early Stage 3 elements, even while Stage 2 work continues.
It introduces the base 'driver.py' and 'feature_schema.py' files.

**Stage 3 characteristics already present:**
- Modular file structure (capture, analysis, ML, export)
- Clear input/output boundaries
- Feature schema as a single source of truth
- Validation layers to prevent silent errors
- Reproducible workflows

**Repo is understandable in:**
- ~1 minute (README)
- ~10 minutes (code)

**What is *not* claimed yet:**
- Full production hardening
- Long-term API stability
- Performance optimisation beyond correctness

This keeps the system usable **without overengineering**.

---

## Stage 4-5 — Speed + Experimentation *(Current)* ⚡

**Goal:** Run many small, controlled experiments quickly to test ideas against reality.

Testing multiple learning paradigms against the same task to expose their failure modes under identical constraints.

**Planned focus:**
- Linear Regression (baseline)
- MLP architecture 
- Genetic Algorithms
- Deep Q-Learning

**Each experiment:**
- Changes one variable
- Logs results
- Compares against baselines

- MLP: Hidden Width (Does more capacity help?), (Does it overfit instantly?)
- GA: Mutation Rate (Does more simulation explore or chaos?), (How sensitive is it?)
- DQL: Reward Weight (Does reward shaping dominate?), (When does it collapse?)

## Stage 5–8 — Iteration with Real World *(Current)* ⚡

**Goal:** Touch reality — users, data, constraints.

This stage creates **engineering maturity**, not better training metrics.
At this stage, learning is no longer offline-only or theoretical.
The *same system* built in Stage 3 and exercised in Stage 4–5 is now
tested directly inside the simulator, frame-by-frame, under real timing constraints.

No new core abstractions are introduced here.
Instead, assumptions are stressed.

**Focus areas:**
- Real-time execution (100Hz loop stability)
- Noisy and imperfect telemetry
- Action clipping, saturation, and deadzones
- Latency between observation → decision → actuation
- Failure modes (spins, stalls, divergence, unsafe actions)

---

[Back to the README](../README.md)
