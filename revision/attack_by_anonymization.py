"""Per-attacker attribute inference results for the 'identifiers retained' synthetic datasets (identifier
comparison), so that Figure 6 can show attacks without vs with anonymization in the original figure style.
The anonymized (AnonymizedFaker) clinical data are identical to the main analysis runs, whose per-attacker
results are in privacy_revised_by_run.csv.
"""
import pandas as pd

from common import MODELS, OUT, ROOT, generate, load_real
from identifier_comparison import identifier_metadata
from privacy_analysis import ATTACKERS, CAT_TARGETS, NUM_TARGETS, attack, attack_applicable, prior_guess


def main(seeds=10):
    full = pd.read_csv(ROOT / "clean_data.csv")
    real = load_real()
    meta = identifier_metadata(real, faker=False)
    n = len(real)
    rows = []
    for seed in range(seeds):
        for m in MODELS:
            synth = generate(full, meta, m, seed, tag="_withids")[real.columns]
            for t in CAT_TARGETS + NUM_TARGETS:
                prior = int(prior_guess(synth, real, real, t).sum())
                for atk in ATTACKERS:
                    if attack_applicable(atk, t):
                        correct = int(attack(synth, real, real, t, atk, seed).sum())
                        rows.append({"Source": m, "Seed": seed, "Attacker": atk, "Target": t, "Correct": correct,
                                     "Prior_correct": prior, "Lift_pp": 100 * (correct - prior) / n})
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "privacy_identifiers_retained_by_run.csv", index=False)


if __name__ == "__main__":
    main()
