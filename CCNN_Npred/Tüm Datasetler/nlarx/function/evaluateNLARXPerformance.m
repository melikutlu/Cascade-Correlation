function [yhat_train, fit_train, yhat_val, fit_val, yhat_train_raw, yhat_val_raw] = evaluateNLARXPerformance(dataTraining, dataValidation, model, norm_stats)
    % EVALUATENLARXPERFORMANCE Evaluate NLARX model on training and validation data
    % Returns predictions (normalized and denormalized) and fit percentages
    
    if nargin < 4
        norm_stats = [];
    end
    
    % Simulate on training data
    yhat_train = sim(model, dataTraining);
    if isa(yhat_train, 'iddata')
        yhat_train = yhat_train.y;
    end
    
    % Simulate on validation data
    yhat_val = sim(model, dataValidation);
    if isa(yhat_val, 'iddata')
        yhat_val = yhat_val.y;
    end
    
    % Denormalize if stats are provided
    if ~isempty(norm_stats) && isfield(norm_stats, 'y_mu') && isfield(norm_stats, 'y_std')
        yhat_train_raw = yhat_train * norm_stats.y_std + norm_stats.y_mu;
        yhat_val_raw = yhat_val * norm_stats.y_std + norm_stats.y_mu;
    else
        yhat_train_raw = yhat_train;
        yhat_val_raw = yhat_val;
    end
    
    % Calculate fit percentage (similar to MATLAB's fit definition)
    % fit = 100 * (1 - (||y - yhat|| / ||y - mean(y)||))
    
    % Training fit
    y_train = dataTraining.y;
    norm_err_train = norm(y_train - yhat_train);
    norm_mean_train = norm(y_train - mean(y_train));
    if norm_mean_train == 0
        fit_train = 0;
    else
        fit_train = 100 * (1 - norm_err_train / norm_mean_train);
    end
    fit_train = max(-inf, min(100, fit_train)); % Clamp to reasonable range
    
    % Validation fit
    y_val = dataValidation.y;
    norm_err_val = norm(y_val - yhat_val);
    norm_mean_val = norm(y_val - mean(y_val));
    if norm_mean_val == 0
        fit_val = 0;
    else
        fit_val = 100 * (1 - norm_err_val / norm_mean_val);
    end
    fit_val = max(-inf, min(100, fit_val)); % Clamp to reasonable range
end
