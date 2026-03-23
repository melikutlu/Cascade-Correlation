function [w_o,mse,info,hFig] = trainOutputLayer_Trajectory(X0,U,T,w_o,W_hidden,g,config,plotColor)
    % plotColor (optional): color used when overlaying loss from subsequent calls
    if nargin < 8 || isempty(plotColor)
        plotColor = 'b';
    end
    w_o = dlarray(w_o);
    X0 = dlarray(X0); U = dlarray(U); T = dlarray(T);
    avgG=[]; avgGSq=[]; it=0;
    maxEpochs = config.model.max_epochs_output;
    loss_hist = zeros(maxEpochs,1);
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    window = max(1, round(config.model.moving_avg_window));
    stopByMavg = false;

    numSamples = size(X0,1);
    batchSize = resolveBatchSize(config, 'batch_size_output', numSamples);

    % Prepare overlay-capable figure: find existing figure by name
    figName = 'Output Loss History';
    existingFig = findall(0, 'Type', 'figure', 'Name', figName);
    if isempty(existingFig)
        hFig = figure('Name', figName, 'Color', 'w');
        figure(hFig);
        ax = gca;
        hold(ax, 'on');
        mainLine = plot(ax, NaN, NaN, 'b-', 'LineWidth', 1.5);
        setappdata(hFig, 'initialLossLine', mainLine);
        % track the cumulative epoch offset so overlays start after previous runs
        setappdata(hFig, 'lossEpochOffset', 0);
        overlayLine = [];
    else
        hFig = existingFig(1);
        figure(hFig);
        ax = gca;
        hold(ax, 'on');
        % read current cumulative offset (defaults to 0)
        offset = getappdata(hFig, 'lossEpochOffset');
        if isempty(offset)
            offset = 0;
            setappdata(hFig, 'lossEpochOffset', offset);
        end
        % create a new overlay line for this call so we can update it during epochs
        overlayLine = plot(ax, NaN, NaN, plotColor, 'LineWidth', 1.5);
        mainLine = getappdata(hFig, 'initialLossLine');
    end

    for ep=1:maxEpochs
        batches = buildMiniBatchOrder(numSamples, batchSize);
        epochLoss = 0;
        for b=1:numel(batches)
            idx = batches{b};
            Xb = X0(idx,:); Ub = U(idx,:); Tb = T(idx,:);
            it = it+1;
            [L,grad] = dlfeval(@loss_output_traj, w_o, Xb, Ub, Tb, W_hidden, g, config);
            
            % Gradient clipping to prevent explosion
            max_grad_norm = 10;
            grad_norm = sqrt(sum(grad.^2));
            if grad_norm > max_grad_norm
                grad = grad * (max_grad_norm / grad_norm);
            end
            
            [w_o, avgG, avgGSq] = adamupdate(w_o, grad, avgG, avgGSq, it, config.model.eta_output);
            batchLoss = gather(L);
            epochLoss = epochLoss + batchLoss;
        end

        epochLoss = epochLoss/numel(batches);
      
        loss_hist(ep) = epochLoss;

        % Update either the overlay line (for retrains) or the main line (first call)
        if ~isempty(overlayLine) && isvalid(overlayLine)
            % overlay should start at offset+1 so it continues after previous data
            if exist('offset','var')
                set(overlayLine, 'XData', offset + (1:ep), 'YData', loss_hist(1:ep));
            else
                set(overlayLine, 'XData', 1:ep, 'YData', loss_hist(1:ep));
            end
        elseif ~isempty(mainLine) && isvalid(mainLine)
            set(mainLine, 'XData', 1:ep, 'YData', loss_hist(1:ep));
        else
            plot(1:ep, loss_hist(1:ep), plotColor, 'LineWidth', 1.5);
        end
        title('Loss History');
        drawnow;

        if config.model.use_plateau_stop && ep > window
            mavg = mean(loss_hist(ep-window:ep-1));
            if mavg - epochLoss <= minDelta
                plateauEpoch = ep;
                stopByMavg = true;
                break;
            end
        end
    end
    % keep w_o as dlarray to avoid repeated host/device transfers
    % (callers can gather explicitly if needed)
    epochs_run = ep;
    loss_hist = loss_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, 'loss_history', loss_hist, 'stop_by_moving_avg', stopByMavg);
    % update cumulative offset so next overlay starts after this run
    if exist('offset','var')
        offset = offset + numel(loss_hist);
        setappdata(hFig, 'lossEpochOffset', offset);
    else
        setappdata(hFig, 'lossEpochOffset', numel(loss_hist));
    end
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    mse = gather(l2loss(Yvec, Tvec, 'DataFormat', 'CB'));
    % return handle to the loss-history figure for external saving/logging
    if exist('hFig','var') && ishandle(hFig)
        % already assigned
    elseif exist('existingFig','var') && ~isempty(existingFig)
        hFig = existingFig(1);
    else
        hFig = [];
    end
end
