# iPrune

Code for *Realized-path state-change certificates for task-conditioned
quantum-circuit pruning*. Local review snapshot: 2026-10-17, manuscript r26. This local package has not been pushed to GitHub.

## Layout
```
iprune/                 geometry, circuit operations and physical pruning
  experiments/          shared task, training and evaluation implementations
  figure_qa/            scientific-figure alignment helpers
scripts/
  experiments/          experiment runners and resource benchmarks
  analysis/             archived-result aggregation and statistical analysis
  figures/              plotting and figure generation
  legacy/               historical sweep entry points, retained for provenance
tests/                  existing regression tests
runs/                   archived experiment JSON records, including pilots
results/                archived additional experiments and test fixtures
```

## Environment
Use Python 3.10+ in an isolated environment, then install:
```
python -m pip install -r requirements.txt
```
For auxiliary figure generation use `requirements-figures.txt`. Some historical
figure utilities depend on Windows fonts. Dependency bounds are not a frozen
clean-environment lockfile. MNIST data are downloaded separately through torchvision.

Run all commands below from the repository root. Use module execution (`-m`)
so shared imports resolve consistently.

## Reproduce manuscript summary tables without training
```
python -m scripts.analysis.summarize_pra_revision_audits
```
This reads bundled records and reconstructs certificate tightness, prefix/state
violation counts, and the multiscale matched-count table. The relevant record
directories are selected explicitly in that script. All archived records are
retained; pilots and superseded protocols are not additional independent evidence.

## Main experiments
| Purpose | Module under `scripts.experiments` |
|---|---|
| TFIM physical sequential deletion | `run_tfim_pruning_phase_seed` |
| Prefix-order mechanism | `run_tfim_order_greedy_ablation_seed` |
| Task-state mechanism | `run_tfim_state_conditioning_ablation_seed` |
| Dynamic objective baseline | `run_tfim_dynamic_objective_baseline` |
| Trained classification | `run_trained_qnn_true_pruning_seed` |
| Molecular energy | `run_h2_qiprune_pilot` |
| QFI gate comparison | `run_qfi_iprune_gatelevel` |
| Fixed-path finite-shot audit | `benchmark_shot_overlap` |
| Selector timing | `benchmark_selection_cost` |

Example argument inspection:
```
python -m scripts.experiments.run_tfim_pruning_phase_seed --help
```
Example small run (not a substitute for the paper's full protocol):
```
python -m scripts.experiments.run_tfim_pruning_phase_seed --seed 3001 --n_qubits 4 --qbeta 0.0 --deltas 0.6 --output_json outputs/tfim_seed3001.json
```
Dynamic objective control on an archived checkpoint:
```
python -m scripts.experiments.run_tfim_dynamic_objective_baseline --source_json runs/tfim_pruning_phase_pilot_20260915/seed3001.json --output_json outputs/dynamic_seed3001.json
```
Use fresh output paths. Optimization may be expensive; do not overwrite archives.

## Protocol boundaries
Algorithm 1 uses q=1. The main multiscale matched-count table retains historical dual-rule records with q=1.030454533953517; both weighted and physical angles must pass. Other historical records are retained for provenance. Do not describe the multiscale table as physical-only q=1. Seeds are independent replicates; gates, thresholds
and synthetic shot trials within a checkpoint are repeated measurements.
The multiscale table uses seeds 4001–4003, N=4,6,8,10 and thresholds 0.6,1.0.
The certificate controls immediate state disturbance on the realized path.
Recovery, distributional generalization and hardware benefit require separate
evidence. Candidate-pool compression in historical scripts is not physical deletion.

## Tests
```
python -m unittest discover -s tests -v
```
The release was checked with the existing tests, archived-summary reconstruction,
and argument parsing for the main runners. Training was not repeated during packaging.

## License and citation
The authors have not yet selected a distribution license. No open-source rights
are implied by this snapshot. Add the approved license and final publication
identifier before public release. Repository: https://github.com/qiprune-quantum/iprune.


## Manuscript experiment records
The table below distinguishes the paper's evidence from the historical and pilot
records also retained for provenance. Numerical archives include JSON, plotting
CSV inputs and the nine saved TFIM benchmark parameter checkpoints in NPZ format.
They do not contain every intermediate state vector from every training step.

| Manuscript result | Archived input under the repository root |
|---|---|
| Physical-only 12-seed TFIM scan and certificate tightness | `runs/tfim_n4_physical_q1_20260924/` |
| Primary matched-count baselines | `runs/formal_tfim_d4_dynamic_q1_20260911/`, `runs/formal_mnist49_dynamic_q1_20260911/` |
| Dynamic sequential objective baseline | `runs/tfim_dynamic_objective_baseline_20261015/` |
| Prefix update and task-state mechanism | `runs/tfim_order_greedy_ablation_20260916/`, `runs/tfim_state_conditioning_ablation_20260916_v2/` |
| QFI comparison and heatmap | `runs/qfi_iprune_gatelevel_n4_20260916/`, `runs/qfi_iprune_gatelevel_n6_20260916/` (including `gate_level_results.csv`) |
| Historical dual-rule multi-size matched-count comparison (q=1.030454533953517) | `runs/tfim_scale_matched_baselines_20260923/`, `runs/tfim_n10_matched_baselines_20260923/` |
| Physical q=1 size and connectivity scans | `runs/tfim_n4_physical_q1_20260924/`, `runs/tfim_n6_physical_q1_20260924/`, `runs/tfim_n8_physical_q1_20260924/`, `runs/tfim_n4_linear_physical_q1_20260924/`, `runs/tfim_n4_alltoall_physical_q1_20260924/` |
| Path-budget and calibration-size controls | `runs/tfim_path_budget_ablation_20260916/`, `runs/tfim_calibration_size_ablation_20260916/`, `runs/iterative_b_mnist_calibration_size_pilot_20260911/` |
| MNIST task-state and held-out audits | `results/task_conditioning_ablation_20260811/mnist49/`, `results/certificate_generalization_20260812/` |
| Molecular energies at five geometries | `runs/h2_vqe_pilot_audited_20260913/`, `runs/h2_short_050A_audited_20260913/`, `runs/h2_mid_100A_audited_20260913/`, `runs/h2_stretched_150A_audited_20260913/`, `runs/h2_far_200A_audited_20260913/` |
| Timing and finite-shot measurement audit | `runs/selection_cost_20260922/` (includes `checkpoints/`, overlap-audit and measurement-budget JSON) |
| Ten-qubit depolarizing-noise test | `runs/tfim_n10_noise_minimal_20260923/` |
| Physical versus historical q-weighted sensitivity | `runs/tfim_n4_physical_q1_20260924/`, `runs/tfim_n6_physical_q1_20260924/`, `runs/tfim_n8_physical_q1_20260924/` and historical phase records; inspect each file's q and acceptance metadata before pooling |

The original scale-study data have been moved into `runs/`; no separate local
lab-bundle directory is needed. Older `results/` subdirectories include candidate-pool
experiments; they must not be substituted for physical deletion results.

### Regenerate scientific figures from archived data
```
python -m scripts.figures.plot_qfi_iprune_predictive
python -m scripts.figures.plot_pruning_heatmap_n4_n6
python -m scripts.figures.plot_tfim_scaling_n4_n6_n8
```
### Reconstruct shot accounting
```
python -m scripts.analysis.analyze_tfim_measurement_budget --audit runs/selection_cost_20260922/shot_overlap_audit_delta_0p5.json --output outputs/measurement_budget.json
```
To repeat the synthetic fixed-path measurement trial using saved parameters:
```
python -m scripts.experiments.benchmark_shot_overlap --root runs/selection_cost_20260922 --delta 0.5 --shots 100 1000 10000 --trials 1000
```
This last command writes an audit file under the specified root. Use a copy of
that directory if you wish to retain the bundled archive byte-for-byte.

### Matched selection timing, including the dynamic objective probe

`runs/selection_cost_dynamic_20261016/` contains a fresh comparison on the saved
TFIM checkpoints (N=4,6,8; seeds 3001-3003; delta=1.2). All three selectors
receive the same realized iPrune deletion count. Each fresh process warms its
selector, then times one selection call and samples peak process RSS. Three
repetitions per checkpoint are summarized by their median. Training and target-count
setup are excluded. These timings compare the current implementations, not
hardware execution or equal-cost oracle primitives.

```
python -m scripts.experiments.benchmark_selection_cost run --root runs/selection_cost_dynamic_20261016 --sizes 4 6 8 --seeds 3001 3002 3003 --delta 1.2 --repeats 3
```

Existing raw outputs are preserved; use a new root with copies of the saved
`checkpoints/` to repeat the benchmark. Historical `qiprune` identifiers and
q-weighted diagnostic labels are retained for archive compatibility. The main
physical selector is iPrune with q=1; historical labels do not imply a demonstrated
advantage from q deformation.

### Implementation provenance

There is no separate Supplemental Material in manuscript r26. Required design
and sampling conventions are given in the main article. The dynamic objective
control retains per-step candidate counts, selected positions and objective
increments; reconstructed checkpoint signatures were checked against the 12
archived unpruned circuits. Multiscale seeds 4001-4003 are distinct from the
12-seed primary TFIM scan; the incompletely covered historical third threshold
is excluded from the multiscale table. Fixed-path shot tests draw 1000 synthetic
Bernoulli trials per checkpoint and shot budget using three independent training
seeds per size. They apply a one-sided tail probability 0.05/L per accepted
overlap and do not implement adaptive shot-based selection.
