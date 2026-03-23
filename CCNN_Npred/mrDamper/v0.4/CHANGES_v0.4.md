# v0.4: Differential Activation Warm-up Phase Implementation

## Problem Statement (Bug)
The system was using differential activation for hidden units ($a(t) = \tanh(z(t) - z(t-1))$), but had a critical initialization bug:

1. `createTrajectoryDataset.m` produced a 1D X0 row vector containing regressors from only a single time step
2. `forwardModelTrajectory.m` initialized the network's previous state ($z_{prev}$) statically from this single X0 row
3. At the first prediction step ($t=1$), the regressors matched the values in X0, causing $z(1) = z(0)$
4. This resulted in $\Delta z(1) = 0$, making the network output zero activation at the first step
5. The network was unable to properly compute derivatives in the cascade correlation training

## Solution: Warm-up Phase Implementation

### Core Concept
Before starting the main N-step prediction loop, the network must process real historical data through its hidden layers to properly initialize the `z_history` (previous state) buffers. This allows the network to compute correct derivatives from the first prediction step onward.

### Changes Made

#### 1. `createTrajectoryDataset.m`
**Change:** X0 now returns a matrix instead of a single row vector
- **Before:** X0 is (Ns × (nu+ny)) - single row of regressors per trajectory
- **After:** X0 is (Ns × (diffOrder+1)×(nu+ny)) - matrix containing the last (diffOrder+1) time steps

**Mechanism:**
- `warmupSteps = diffOrder + 1`
  - For diff-tanh (diffOrder=1): warmupSteps=2, so X0 contains 2 previous time steps
  - For diff2-tanh (diffOrder=2): warmupSteps=3, so X0 contains 3 previous time steps
- For trajectory starting at time $t=1$, X0 contains regressors from times $t=-(warmupSteps-1), ..., -1, 0$
- This ensures the network has actual historical data to feed through before predictions

#### 2. `forwardModelTrajectory.m`
**Change:** Added a warm-up phase before the main prediction loop
- **Before:** z_history was initialized with a static estimate from a single X0 row
- **After:** Two distinct phases:

**Warm-up Phase (Lines ~35-80):**
```
For each warmup step in X0:
  - Extract that step's regressor values
  - Feed through the network (u inputs, y history updates)
  - Compute hidden layer pre-activations z
  - Compute activations using differential mode (a = tanh(z - z_prev))
  - Update z_history buffers (shift and store new z values)
  - Do NOT record any output predictions during warm-up
```

**Main Prediction Phase (Lines ~85-130):**
```
For t = 1 to N:
  - z_history is now properly filled with real historical data
  - Compute activations with correct derivative differences
  - Update y_history with prediction feedback
```

**Key Benefits:**
- Network has real state history before first prediction
- First differential computation $\Delta z(1) = z(1) - z(0)$ is meaningful
- Cascade connection activations are computed correctly from the start

#### 3. `candidateCorrelationMetric.m`
**Change:** Added identical warm-up phase
- **Purpose:** Candidate unit training must operate under the same conditions as the main network
- **Implementation:** Same warm-up loop (Lines ~33-65) before the main N-step correlation computation
- **Consistency:** Ensures z_prev_hidden and z_prev_cand are properly initialized before computing v(M×N)

#### 4. `Npred_MiniBatch_Adam_maxCandidate.m` (Main Training Script)
**Change:** Fixed w_o initialization
- **Before:** `d0 = size(X0_tr, 2)` - incorrect because X0 is now wider (warmupSteps × (nu+ny))
- **After:** `d0 = nu + ny` - explicit calculation from config regressors
- **Reason:** w_o should match the input layer size (u + y regressors), not the entire X0 width

## Mathematical Guarantee

With the warm-up phase:
- At $t=1$, both $z(1)$ and $z(0)$ come from actual network evaluations on real data
- $\Delta z = z(1) - z(0) \neq 0$ (genuine difference, not forced zero)
- Gradient flow through differential activation is meaningful from the first prediction step
- Cascade correlation training can properly select candidate units that improve predictions

## Files Modified
1. `function/createTrajectoryDataset.m` - X0 structure changed from 1D to 3D (with warmup window)
2. `function/forwardModelTrajectory.m` - Added warm-up phase (lines 30-80)
3. `function/candidateCorrelationMetric.m` - Added warm-up phase (lines 34-66)
4. `Npred_MiniBatch_Adam_maxCandidate.m` - Fixed w_o initialization (lines 96-99)

## Testing Recommendations
1. Verify that X0 dimensions are as expected: (Ns, warmupSteps×(nu+ny))
2. Check that first predictions (t=1) now have non-zero activations in diff-tanh mode
3. Compare network performance with v0.3 - should see improved convergence
4. Validate that cascade correlation training selects better candidate units
5. Check that gradient flow is no longer blocked at the first prediction step

## Backward Compatibility Notes
- X0 structure change is NOT compatible with code expecting 2D X0
- Any custom evaluation scripts must be updated to handle 3D X0
- The configuration structure remains unchanged
- Regressor definitions (u, y lags) work the same way

## Version Information
- **Version:** v0.4
- **Base:** v0.3_deneme_diff
- **Date:** 2026-03-19
- **Change Type:** Bug fix (differential activation initialization)
