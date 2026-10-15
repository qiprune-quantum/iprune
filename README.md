# iPrune

Code for *Realized-path state-change certificates for task-conditioned
quantum-circuit pruning*. Snapshot: 2026-10-15, manuscript r5.

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
results/                historical checkpoint fixtures required by tests
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
python -m scripts.experiments.run_tfim_pruning_phase_seed --seed 3001 --n_qubits 4 --deltas 0.6 --output_json outputs/tfim_seed3001.json
```
Dynamic objective control on an archived checkpoint:
```
python -m scripts.experiments.run_tfim_dynamic_objective_baseline --source_json runs/tfim_pruning_phase_pilot_20260915/seed3001.json --output_json outputs/dynamic_seed3001.json
```
Use fresh output paths. Optimization may be expensive; do not overwrite archives.

## Protocol boundaries
The physical selector uses q=1. Historical weighted/dual rules are separate
sensitivity experiments. Seeds are independent replicates; gates, thresholds
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
are implied by this snapshot. Add the approved license, real repository URL and
final publication identifier before public release.
