clear; clc; close all;

%% ===================== CONFIG =====================
config.data.source = 'robotarmdata';   % 'twotankdata', 'dryer2', 'mrdamper', 'robotarmdata'
config.data.train_ratio = 0.5;
config.data.val_ratio   = 0.5;

% twotank defaults
config.data.twotank.warmup_samples = 20;
config.data.twotank.filter_cutoff  = 0;      % Hz, 0 => no filter
config.data.twotank.sampling_time  = 0.2;

% dryer2 defaults
config.data.dryer2.sampling_time   = 0.08;

% mrdamper defaults
config.data.mrdamper.sampling_time = 0.01;   % change if needed

% robotarm defaults
config.data.robotarm.validation_experiment = 1;       % 1, 2, or 3
config.data.robotarm.original_sampling_time = 0.001;  % change if needed
config.data.robotarm.downsample_factor = 1;           % positive integer

config.modelType = 'ss';           % 'tf' or 'ss'
config.mode      = 'simulation';      % 'predict' or 'simulation'

config.tf_num_order = 5;          % numerator order
config.tf_den_order = 5;          % denominator order
config.tf_io_delay  = 0;           % input delay

config.ss_order = 4;

config.makePlots = true;
config.savePlots = true;
config.logFolder = 'log';

%% ===================== RUN =====================
try
    [Utr, Ytr, Uva, Yva, Ts, dataInfo] = loadDatasetSimple(config);

    results = processModel(Utr, Ytr, Uva, Yva, Ts, config, dataInfo);

    results = writeLog(results, config, dataInfo);

    fprintf('\n============================================\n');
    fprintf('Run completed successfully.\n');
    fprintf('Dataset     : %s\n', dataInfo.name);
    fprintf('Model type  : %s\n', upper(config.modelType));
    fprintf('Mode        : %s\n', lower(config.mode));
    fprintf('RMSE        : %.6f\n', results.rmse);
    fprintf('Log folder  : %s\n', results.runFolder);
    fprintf('============================================\n\n');

catch ME
    fprintf(2, '\nERROR: %s\n', ME.message);
    rethrow(ME);
end