function [w_o, mse, info, hFig] = trainOutputLayer_TrajectorySimPlateau(X0, U, T, Ufull, Yfull, w_o, W_hidden, g, config, plotColor)
    % Train the output layer in fixed epoch blocks and stop when the
    % recursive simulation loss no longer improves over the previous two blocks.
    if nargin < 10 || isempty(plotColor)
        plotColor = 'b';
    end

    blockEpochs = 20;
    if isfield(config, 'model') && isfield(config.model, 'max_epochs_output') && ~isempty(config.model.max_epochs_output)
        blockEpochs = config.model.max_epochs_output;
    end

    maxBlocks = 50;
    if isfield(config, 'model') && isfield(config.model, 'max_output_blocks') && ~isempty(config.model.max_output_blocks)
        maxBlocks = config.model.max_output_blocks;
    end
    maxBlocks = max(maxBlocks, 3);

    totalEpochs = 0;
    blockEpochHistory = zeros(0, 1);
    simLossHistory = zeros(0, 1);
    lossHistoryHistory = cell(0, 1);
    stopBySimPlateau = false;
    plateauEpoch = NaN;
    hFig = [];

    for blockIdx = 1:maxBlocks
        blockConfig = config;
        blockConfig.model.max_epochs_output = blockEpochs;
        blockConfig.model.use_plateau_stop = false;

        [w_o, ~, blockInfo, hFig] = trainOutputLayer_Trajectory(X0, U, T, w_o, W_hidden, g, blockConfig, plotColor);

        if isfield(blockInfo, 'epochs_run') && ~isempty(blockInfo.epochs_run)
            blockEpochHistory(end+1, 1) = blockInfo.epochs_run; %#ok<AGROW>
            totalEpochs = totalEpochs + blockInfo.epochs_run;
        else
            blockEpochHistory(end+1, 1) = blockEpochs; %#ok<AGROW>
            totalEpochs = totalEpochs + blockEpochs;
        end

        if isfield(blockInfo, 'loss_history') && ~isempty(blockInfo.loss_history)
            lossHistoryHistory{end+1, 1} = blockInfo.loss_history(:); %#ok<AGROW>
        end

        Yhat_sim = recursivePredictFullSeries(Ufull, Yfull, W_hidden, w_o, g, config);
        simLoss = mean((Yfull(2:end) - Yhat_sim(2:end)).^2);
        simLossHistory(end+1, 1) = simLoss; %#ok<AGROW>

        if ~isfinite(simLoss)
            stopBySimPlateau = true;
            plateauEpoch = totalEpochs;
            break;
        end

        if numel(simLossHistory) >= 3
            prevAvg = mean(simLossHistory(end-2:end-1));
            if simLoss >= prevAvg
                stopBySimPlateau = true;
                plateauEpoch = totalEpochs;
                break;
            end
        end
    end

    if isempty(simLossHistory)
        mse = NaN;
    else
        mse = simLossHistory(end);
    end

    if isempty(lossHistoryHistory)
        lossHistory = [];
    else
        lossHistory = vertcat(lossHistoryHistory{:});
    end

    info = struct();
    info.epochs_run = totalEpochs;
    info.plateau_epoch = plateauEpoch;
    info.block_count = numel(simLossHistory);
    info.block_epochs_run = blockEpochHistory;
    info.sim_loss_history = simLossHistory;
    info.loss_history = lossHistory;
    info.stop_by_sim_plateau = stopBySimPlateau;
    info.stop_by_moving_avg = false;
end