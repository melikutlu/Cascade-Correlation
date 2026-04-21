function [w_o, mse, info, hFig] = trainOutputLayer_TrajectorySimPlateau(X0, U, T, Ufull, Yfull, w_o, W_hidden, g, config, plotColor)
    % Train the output layer in epoch blocks, run a final short block for
    % any remainder, and stop when the recursive simulation loss no longer
    % improves over the previous two blocks.
    hiddenCount = numel(W_hidden);
    if mod(hiddenCount, 2) == 0
        plotColor = [0.0000, 0.4470, 0.7410]; % blue
    else
        plotColor = [1.0000, 0.4078, 0.7059]; % pink
    end

    blockEpochs = 20;
    if isfield(config, 'model') && isfield(config.model, 'sim_loss_eval_interval') && ~isempty(config.model.sim_loss_eval_interval)
        blockEpochs = config.model.sim_loss_eval_interval;
    elseif isfield(config, 'model') && isfield(config.model, 'max_epochs_output') && ~isempty(config.model.max_epochs_output)
        blockEpochs = config.model.max_epochs_output;
    end
    blockEpochs = max(1, round(blockEpochs));

    totalEpochBudget = blockEpochs * 50;
    if isfield(config, 'model') && isfield(config.model, 'output_max_epochs') && ~isempty(config.model.output_max_epochs)
        totalEpochBudget = config.model.output_max_epochs;
    end
    totalEpochBudget = max(1, round(totalEpochBudget));

    minBlocks = 3;
    if isfield(config, 'model') && isfield(config.model, 'sim_loss_min_blocks') && ~isempty(config.model.sim_loss_min_blocks)
        minBlocks = config.model.sim_loss_min_blocks;
    end
    minBlocks = max(3, round(minBlocks));
    maxBlocksBudget = max(1, ceil(totalEpochBudget / blockEpochs));
    minBlocksRequested = minBlocks;
    minBlocks = min(minBlocks, maxBlocksBudget);
    minBlocksClamped = minBlocks < minBlocksRequested;
    forceFullEpochs = isfield(config, 'model') && isfield(config.model, 'force_output_full_epochs') && config.model.force_output_full_epochs;

    if minBlocksClamped
        warning('trainOutputLayer_TrajectorySimPlateau:MinBlocksClamped', ...
            'Requested sim_loss_min_blocks=%d exceeds budget-derived max blocks=%d. Using %d blocks instead.', ...
            minBlocksRequested, maxBlocksBudget, minBlocks);
    end

    totalEpochs = 0;
    remainingEpochBudget = totalEpochBudget;
    blockEpochHistory = zeros(0, 1);
    simLossHistory = zeros(0, 1);
    lossHistoryHistory = cell(0, 1);
    stopBySimPlateau = false;
    plateauEpoch = NaN;
    hFig = [];

    if forceFullEpochs
        blockConfig = config;
        blockConfig.model.max_epochs_output = totalEpochBudget;
        blockConfig.model.use_plateau_stop = false;

        [w_o, ~, blockInfo, hFig] = trainOutputLayer_Trajectory(X0, U, T, w_o, W_hidden, g, blockConfig, plotColor);

        epochsThisBlock = totalEpochBudget;
        if isfield(blockInfo, 'epochs_run') && ~isempty(blockInfo.epochs_run)
            epochsThisBlock = blockInfo.epochs_run;
        end

        epochsThisBlock = max(0, round(epochsThisBlock));
        blockEpochHistory(end+1, 1) = epochsThisBlock; %#ok<AGROW>
        totalEpochs = totalEpochs + epochsThisBlock;
        remainingEpochBudget = max(0, remainingEpochBudget - epochsThisBlock);

        if isfield(blockInfo, 'loss_history') && ~isempty(blockInfo.loss_history)
            lossHistoryHistory{end+1, 1} = blockInfo.loss_history(:); %#ok<AGROW>
        end

        Yhat_sim = recursivePredictFullSeries(Ufull, Yfull, W_hidden, w_o, g, config);
        simLoss = mean((Yfull(2:end) - Yhat_sim(2:end)).^2);
        simLossHistory(end+1, 1) = simLoss; %#ok<AGROW>

        if ~isfinite(simLoss)
            stopBySimPlateau = true;
            plateauEpoch = totalEpochs;
        end
    else
        while remainingEpochBudget > 0
            currentBlockEpochs = min(blockEpochs, remainingEpochBudget);
            blockConfig = config;
            blockConfig.model.max_epochs_output = currentBlockEpochs;
            blockConfig.model.use_plateau_stop = false;

            [w_o, ~, blockInfo, hFig] = trainOutputLayer_Trajectory(X0, U, T, w_o, W_hidden, g, blockConfig, plotColor);

            epochsThisBlock = currentBlockEpochs;
            if isfield(blockInfo, 'epochs_run') && ~isempty(blockInfo.epochs_run)
                epochsThisBlock = blockInfo.epochs_run;
            end

            epochsThisBlock = max(0, round(epochsThisBlock));
            blockEpochHistory(end+1, 1) = epochsThisBlock; %#ok<AGROW>
            totalEpochs = totalEpochs + epochsThisBlock;
            remainingEpochBudget = max(0, remainingEpochBudget - epochsThisBlock);

            if epochsThisBlock <= 0
                stopBySimPlateau = true;
                plateauEpoch = totalEpochs;
                break;
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

            if numel(simLossHistory) >= minBlocks
                prevAvg = mean(simLossHistory(end-2:end-1));
                if simLoss >= prevAvg
                    stopBySimPlateau = true;
                    plateauEpoch = totalEpochs;
                    break;
                end
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
    info.sim_loss_min_blocks_requested = minBlocksRequested;
    info.sim_loss_min_blocks_effective = minBlocks;
    info.sim_loss_min_blocks_clamped = minBlocksClamped;
    info.output_max_blocks_budget = maxBlocksBudget;
end