---
type: device-twin
id: synthetic-bosonic-rehearsal
family: bosonic
platform: bosonic
status: seed
route_intent: team
date: 2026-09-02
parameters:
  chi_kHz: -300.0          # dispersive shift χ/2π (synthetic, record-class magnitude)
  K_q_GHz: -0.161          # transmon anharmonicity/2π (class shape; structurally zero on 2 levels)
  K_c_kHz: -82.0           # cavity Kerr/2π (synthetic)
  chi_p_kHz: 0.0           # χ′ — unmeasured; held at 0
  N_transmon: 2            # qubit levels in the model
  N_fock: 4                # cavity Fock cutoff (rehearsal-sized: contains the prep's support)
noise:
  T1_q_us: {value: 120.0, estimate: true, note: "synthetic; unused by the rehearsal's closed-system family slice (decay rides the substrate's OpenQuantumSystem dispatch, post-0.3.0)"}
  kappa_c_per_us: {value: 0.0008, estimate: true, note: "synthetic; unused by the closed-system family slice — see T1 note"}
  readout_confusion: {value: [[0.97, 0.03], [0.06, 0.94]], estimate: true, note: "synthetic placeholder matrix (2×2 against the transmon-ancilla marginal)"}
drift_priors:
  chi_kHz:
    process: [ou, ramp]
    sigma_rel: 0.01
    tau_days: 10
    ramp_kHz_per_day: -3.0
provenance:
  source: "Intonato M3b rehearsal fixture — bosonic seed-family schema shape (mirrors Strumento's synthetic-bosonic fixture)"
  measured: 2026-09-02
  note: "All values synthetic; no device behind them. The χ prior composes the record family's OU shape (σ_rel, τ) with a class-realistic thermal-soak Ramp at the prior's own scale a 3 kHz/day thermal soak declared at rehearsal scale (the OU's mean-reversion toward the record value fights the soak — their steady-state tension sets the rehearsal's ~30 kHz drift excursion; σ_rel is the class-typical 0.01, cf. the Strumento bosonic fixture's 0.008). The soak is negative — the rehearsal's structured prep is fragile for χ drifting more negative (its comb pulse's detuning geometry), and the drift must exercise the fragile sector the loop has to track — the drift-process catalogue's monotone component — so the rehearsal's adapted-vs-frozen delta is unambiguous over its 3 epochs. The rehearsal harness builds its DriftPlan from THESE priors."
tags: [device-twin, bosonic, fixture, m3b-rehearsal]
---

# Synthetic bosonic twin — M3b rehearsal fixture

The QILC-through-twin rehearsal's twin record (#37): transmon ancilla dispersively
coupled to a storage cavity, sized for the rehearsal (4 Fock levels contain the
Ramsey prep's photon support). Every value is synthetic — this record exists to
give the rehearsal a schema-faithful, committed twin; the vault record governs
the real campaign.
