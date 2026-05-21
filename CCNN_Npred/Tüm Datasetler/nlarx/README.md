# NLARX Training Framework

This folder contains MATLAB scripts for training Nonlinear AutoRegressive with eXogenous input (NLARX) models using cascade-correlation neural networks with sigmoid activation.

## Folder Structure

```
nlarx/
├── NLARX_MainScript.m          # Main training script (entry point)
├── function/
│   ├── loadDataForNLARX.m      # Load datasets (twotankdata, dryer2, mrdamper)
│   ├── evaluateNLARXPerformance.m   # Evaluate model performance
│   ├── calculateRMSE.m         # Calculate RMSE metric
│   └── writeNLARXLog.m         # Write training results to log files
└── logs/                        # Training logs (created automatically)
    ├── twotankdata_CV/
    ├── twotankdata_NoCV/
    ├── dryer2_CV/
    ├── dryer2_NoCV/
    ├── mrdamper_CV/
    └── mrdamper_NoCV/
```

## Supported Datasets

The framework supports three datasets:

1. **twotankdata** - Two-tank system
   - Input: u (single input)
   - Output: y (tank level)
   - Default sampling time: 1 second

2. **dryer2** - Industrial dryer
   - Input: u2 (control signal)
   - Output: y2 (temperature/moisture)
   - Default sampling time: 0.08 seconds

3. **mrdamper** - Magnetorheological damper
   - Input: Velocity (V)
   - Output: Force (F)
   - Default sampling time: 0.01 seconds

## Usage

### Quick Start

1. Open MATLAB and navigate to the `nlarx` folder
2. Run the main script:
   ```matlab
   NLARX_MainScript
   ```
3. Select dataset when prompted (1-3)
4. The script will:
   - Load the selected dataset
   - Train two models (WITH and WITHOUT cross-validation)
   - Display comparison plots
   - Save results to logs folder

### Running Specific Datasets

To run a specific dataset without interactive selection, modify `NLARX_MainScript.m`:

```matlab
% Instead of interactive selection, directly set:
datasetName = 'dryer2';  % or 'twotankdata', 'mrdamper'
[dataTraining, dataValidation, dataInfo] = loadDataForNLARX(datasetName);
```

## Model Configuration

Key parameters in `NLARX_MainScript.m`:

```matlab
% Neural network setup
activation = 'sigmoid';           % Activation function
maxHiddenUnits = 20;             % Maximum hidden units for cascade-correlation

% Regressor orders [na, nb, nk]
orders = [1, 1, 1];              % Output lag, Input lag, Dead time
% - na = number of output lags (y(t-1), y(t-2), ...)
% - nb = number of input lags (u(t-1), u(t-2), ...)
% - nk = input delay (dead time)
```

## Performance Metrics

The script calculates and logs:

- **Training Fit (%)**: Fit percentage on training data
- **Training RMSE**: Root Mean Squared Error on training data
- **Validation Fit (%)**: Fit percentage on validation data
- **Validation RMSE**: Root Mean Squared Error on validation data

## Log Files

Training results are saved as:
- **Text log** (`.txt`): Human-readable results
- **MAT file** (`.mat`): Complete training information structure

Logs are organized by dataset and training method:
- `logs/twotankdata_CV/` - Two-tank with cross-validation
- `logs/twotankdata_NoCV/` - Two-tank without cross-validation
- `logs/dryer2_CV/` - Dryer with cross-validation
- `logs/dryer2_NoCV/` - Dryer without cross-validation
- `logs/mrdamper_CV/` - MR Damper with cross-validation
- `logs/mrdamper_NoCV/` - MR Damper without cross-validation

## Output Figures

The script generates the following figures:

1. **Input Data Visualization** - Training and validation input signals
2. **Training Fit (WITH CV)** - Predicted vs actual on training data
3. **Validation Fit (WITH CV)** - Predicted vs actual on validation data
4. **Training Fit (WITHOUT CV)** - Predicted vs actual on training data
5. **Validation Fit (WITHOUT CV)** - Predicted vs actual on validation data

## Comparison: WITH vs WITHOUT Cross-Validation

The script trains two models to compare training strategies:

- **WITH Cross-Validation**: Uses 10% of training data for validation during training
  - May result in better generalization
  - Training is typically slower
  
- **WITHOUT Cross-Validation**: Uses all training data for fitting
  - May achieve better training performance
  - Higher risk of overfitting

## Adding New Datasets

To add a new dataset, modify `loadDataForNLARX.m`:

1. Add a new case in the switch statement
2. Load your data (must contain input `u` and output `y`)
3. Create `iddata` objects:
   ```matlab
   dataTraining = iddata(yTr, uTr, Ts, 'OutputName', 'output', 'InputName', 'input');
   dataValidation = iddata(yVa, uVa, Ts, 'OutputName', 'output', 'InputName', 'input');
   ```
4. Update `dataInfo` structure

## Notes

- All data is automatically split into training (50%) and validation (50%) subsets
- Data is loaded from the parent workspace directory
- The cascade-correlation architecture automatically grows hidden units
- Sigmoid activation function is used for hidden neurons
- Results can be analyzed separately for each dataset

## MATLAB Requirements

- System Identification Toolbox (for iddata, nlarx)
- Neural Network Toolbox (for idNeuralNetwork)

---

For performance analysis and comparison across datasets, refer to the log files in the `logs` folder.
