function [w_h, best_metric, info] = trainCandidateUnit_Corr(X0,U,T,W_hidden,w_o,g,config)
    % Train a candidate unit to MAXIMIZE correlation^2 with the N-step residual using mini-batches.

    % X0 shape: (Ns, nWarmupSteps, nu+ny)
    % Extract feature dimension (last dimension)
    nFeatures = size(X0,3);
    d = nFeatures + numel(W_hidden); % candidate input dim
    w_h = dlarray(randn(d,1)*0.01);

    X0_d = dlarray(X0); 
    U_d = dlarray(U); 
    T_d = dlarray(T);
    w_o_d = dlarray(w_o);

    avgG=[]; 
    avgGSq=[]; 
    it=0; 
    best_metric = 0;  %algoritmanın ürettiği ilk skor kötü olsa bile devam edebilmesi için
    best_w = w_h;
    maxEpochs = config.model.max_epochs_candidate;
    metric_hist = zeros(maxEpochs,1);
    w_h_norm_hist = zeros(maxEpochs,1);
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    window = max(1, round(config.model.moving_avg_window));

    numSamples = size(X0,1);
    batchSize = resolveBatchSize(config, 'batch_size_candidate', numSamples);

    for ep=1:maxEpochs
        batches = buildMiniBatchOrder(numSamples, batchSize);
        for b=1:numel(batches)
            idx = batches{b};
            Xb = X0_d(idx,:,:); 
            Ub = U_d(idx,:); 
            Tb = T_d(idx,:);
            it = it + 1;
            [loss, ~, grad] = dlfeval(@loss_candidate_corr, w_h, Xb, Ub, Tb, W_hidden, w_o_d, g, config);

          
            % Gradient clipping to prevent explosion
            max_grad_norm = 10;
            grad_norm = sqrt(sum(grad.^2));
            if grad_norm > max_grad_norm
                grad = grad * (max_grad_norm / grad_norm);
            end

            [w_h, avgG, avgGSq] = adamupdate(w_h, grad, avgG, avgGSq, it, config.model.eta_candidate);
        end

        metricVal = evaluateCandidateMetric(w_h, X0_d, U_d, T_d, W_hidden, w_o_d, g, config);
        metric_hist(ep) = metricVal;
        w_h_norm_hist(ep) = sqrt(sum(extractdata(w_h).^2));
        
        if metricVal > best_metric
            best_metric = metricVal;
            best_w = w_h;
        end


        if config.model.use_plateau_stop && ep > window
            mavg = mean(metric_hist(ep-window:ep-1));
            if metricVal - mavg <= minDelta
                plateauEpoch = ep;
                break;
            end
        end
    end

    w_h = best_w;
    epochs_run = ep;
    metric_hist = metric_hist(1:epochs_run);
    w_h_norm_hist = w_h_norm_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, 'metric_history', metric_hist, 'w_h_norm_history', w_h_norm_hist);
end
