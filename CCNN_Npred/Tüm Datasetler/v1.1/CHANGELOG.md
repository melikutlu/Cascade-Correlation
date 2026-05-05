# mrDamper v1.0 Change Log

Date: 2026-05-05

## Added / Changed
- Updated `Npred_MiniBatch_Adam_maxCandidate.m` so every accepted hidden stage now records train and validation MSE, RMSE, and fit values.
- Kept hidden growth decisions on the train-side acceptance logic; only the final model selection now chooses the best validation stage.
- Added validation-selected final model metadata and per-stage histories to the parameter log output.
- Added an on/off activation clipping flag so clipping can be disabled globally, and made the clipping logic apply only to the final activation output for tanh and diff-style activations when enabled.

Date: 2026-04-14

## Added / Changed
- Copied the baseline from `mrDamper/v0.8_diff_cliping` into `mrDamper/v1.0`.
- Generalized the main script so the dataset is selected only through `config.data.source`, with dataset-specific defaults applied automatically.
- Added automatic dataset presets for `twotankdata`, `dryer2`, and `mrdamper`.
- Updated `loadDataByConfig_min.m` to canonicalize dataset aliases and apply the correct preprocessing for each dataset.
- Changed run logging so files are stored under `logs/<dataset>/fitTr..._fitVa.../` instead of a generic log folder.
- Simplified the run folder name so it only reflects training and validation fit.
- Added the dataset name at the top of the log file.
- Added configurable diff clipping bounds to the main script:
  - `config.model.diff_clip_lower`
  - `config.model.diff_clip_upper`
- Added configurable hidden-growth controls:
  - `config.model.hidden_bootstrap_count`
  - `config.model.hidden_acceptance_window`
- Added configurable simulation loss interval:
  - `config.model.sim_loss_eval_interval`
- Added configurable minimum simulation-loss blocks:
  - `config.model.sim_loss_min_blocks`
- Added configurable total output-epoch budget:
  - `config.model.output_max_epochs`
- Updated `applyHiddenActivation.m` so both `diff` and `diff-tanh` use the same configurable clipping limits.
- Added `trainOutputLayer_TrajectorySimPlateau.m` to train the output layer in configurable blocks and evaluate recursive simulation loss after each block.
- Updated the block scheduler so any leftover epochs are executed as one final short block instead of being dropped.
- Added a warning and `info` metadata when the requested minimum block count exceeds the budget-derived maximum block count.
- Changed the output-layer stop rule so training stops when the third block loss is not better than the average of the first two block losses.
- Changed hidden acceptance so the first bootstrap hidden units are always accepted, then later hidden units are accepted only if their full recursive MSE beats the rolling mean of the last accepted hidden models.
- Removed the post-hidden NaN/Inf and finite-MSE growth stops from the hidden acceptance block.
- Parameter logs now include the resolved data source and dataset-specific defaults where available.
- Parameter logs now include the diff clipping bounds used during the run.
- Parameter logs now include the hidden-growth controls used during the run:
  - `config.model.hidden_bootstrap_count`
  - `config.model.hidden_acceptance_window`
- Loss visualizations now keep two hidden-unit histories:
  - the final accepted model
  - the pre-revert state, if growth is reverted back to the baseline
- Kept candidate training on the existing plateau / moving-average logic.

## Default Values
- `config.model.max_epochs_output = 20`
- `config.model.max_output_blocks = 50`
- `config.model.diff_clip_lower = -10`
- `config.model.diff_clip_upper = 10`
- `config.model.hidden_bootstrap_count = 5`
- `config.model.hidden_acceptance_window = 5`
- `config.model.sim_loss_eval_interval = 20`
- `config.model.sim_loss_min_blocks = 3`
- `config.model.output_max_epochs = 1000`
- If `sim_loss_min_blocks` is larger than the budget-derived block count, the code clamps it and emits a warning.

## Notes
- To select another dataset, change only `config.data.source` in `Npred_MiniBatch_Adam_maxCandidate.m`.
- Supported dataset names: `twotankdata`, `dryer2`, `mrdamper`.
- To make clipping wider, change the config values in `Npred_MiniBatch_Adam_maxCandidate.m`.
- The new output-layer logic is specific to `v1.0`.
