function results = processModel(Utr, Ytr, Uva, Yva, Ts, config, dataInfo)

    if ~license('test', 'Identification_Toolbox')
        error('System Identification Toolbox is required.');
    end

    zTrain = iddata(Ytr, Utr, Ts);
    zVal   = iddata(Yva, Uva, Ts);

    results = struct();
    results.dataInfo   = dataInfo;
    results.config     = config;
    results.Ts         = Ts;
    results.model      = [];
    results.modelText  = '';
    results.yhat       = [];
    results.rmse       = NaN;
    results.mse        = NaN;
    results.mae        = NaN;
    results.fit        = NaN;
    results.figPaths   = {};
    results.timestamp  = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

    fprintf('\n--------------------------------------------\n');
    fprintf('DATASET SUMMARY\n');
    fprintf('--------------------------------------------\n');
    fprintf('Dataset         : %s\n', dataInfo.name);
    fprintf('Sampling time   : %.6f\n', Ts);
    fprintf('Train samples   : %d\n', dataInfo.Ntrain);
    fprintf('Val samples     : %d\n', dataInfo.Nval);

    %% Estimate model
    switch lower(config.modelType)

        case 'tf'
            fprintf('\n--------------------------------------------\n');
            fprintf('TRANSFER FUNCTION MODEL\n');
            fprintf('--------------------------------------------\n');

            np = config.tf_den_order;
            nz = config.tf_num_order;
            nk = config.tf_io_delay;

            model = tfest(zTrain, np, nz, nk);

        case 'ss'
            fprintf('\n--------------------------------------------\n');
            fprintf('STATE-SPACE MODEL\n');
            fprintf('--------------------------------------------\n');

            nx = config.ss_order;
            model = ssest(zTrain, nx);

        otherwise
            error('Unknown modelType: %s', config.modelType);
    end

    results.model = model;
    disp(model);
    results.modelText = evalc('disp(model)');

    %% Run selected mode
    switch lower(config.mode)

        case 'predict'
            yhat_raw = predict(model, zVal);

        case 'simulation'
            yhat_raw = sim(model, zVal.u);

        otherwise
            error('Unknown mode: %s', config.mode);
    end

    %% Robust output extraction
    if isa(yhat_raw, 'iddata')
        yhat = yhat_raw.y;
    elseif isnumeric(yhat_raw)
        yhat = yhat_raw;
    else
        try
            yhat = yhat_raw.y;
        catch
            error('Unsupported output type returned from predict/sim.');
        end
    end

    ytrue = Yva(:);
    yhat  = yhat(:);

    if length(ytrue) ~= length(yhat)
        m = min(length(ytrue), length(yhat));
        ytrue = ytrue(1:m);
        yhat  = yhat(1:m);
    end

    err = ytrue - yhat;

    results.ytrue = ytrue;
    results.yhat  = yhat;
    results.error = err;

    results.mse  = mean(err.^2);
    results.rmse = sqrt(results.mse);
    results.mae  = mean(abs(err));

    denom = norm(ytrue - mean(ytrue));
    if denom > 0
        results.fit = 100 * (1 - norm(err) / denom);
    else
        results.fit = NaN;
    end

    fprintf('\n--------------------------------------------\n');
    fprintf('VALIDATION RESULTS\n');
    fprintf('--------------------------------------------\n');
    fprintf('Mode            : %s\n', lower(config.mode));
    fprintf('RMSE            : %.6f\n', results.rmse);
    fprintf('MSE             : %.6f\n', results.mse);
    fprintf('MAE             : %.6f\n', results.mae);
    fprintf('Fit (%%)         : %.4f\n', results.fit);

    %% Plot
    if config.makePlots
        f1 = figure('Name', 'Output Comparison');
        plot(ytrue, 'LineWidth', 1.2); hold on;
        plot(yhat,  'LineWidth', 1.2);
        grid on;
        xlabel('Sample');
        ylabel('Output');
        title(sprintf('%s | %s | %s', dataInfo.name, upper(config.modelType), lower(config.mode)));
        legend('Real Output', 'Model Output', 'Location', 'best');

        f2 = figure('Name', 'Error Signal');
        plot(err, 'LineWidth', 1.2);
        grid on;
        xlabel('Sample');
        ylabel('Error');
        title('Prediction / Simulation Error');

        results.figHandles = [f1, f2];
    else
        results.figHandles = [];
    end
end