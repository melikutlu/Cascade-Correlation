function config = applyDatasetDefaults(config)
    if ~isfield(config, 'data') || ~isfield(config.data, 'source') || isempty(config.data.source)
        error('config.data.source must be set to a supported dataset name.');
    end

    source = lower(strtrim(string(config.data.source)));

    switch source
        case {"twotankdata", "twotank", "two_tank", "twotankdataset"}
            config.data.source_label = 'twotankdata';
            if ~isfield(config.data, 'twotank') || isempty(config.data.twotank)
                config.data.twotank = struct();
            end
            config.data.twotank = setDefaultField(config.data.twotank, 'filter_cutoff', 0.066902);
            config.data.twotank = setDefaultField(config.data.twotank, 'warmup_samples', 20);
            config.data.twotank = setDefaultField(config.data.twotank, 'sampling_time', 0.2);
        case {"dryer2", "dryer", "dryerdataset"}
            config.data.source_label = 'dryer2';
            if ~isfield(config.data, 'dryer2') || isempty(config.data.dryer2)
                config.data.dryer2 = struct();
            end
            config.data.dryer2 = setDefaultField(config.data.dryer2, 'sampling_time', 0.08);
        case {"mrdamper", "mr_damper", "mrdamperdataset", "mrdamperdata"}
            config.data.source_label = 'mrDamper';
            if ~isfield(config.data, 'mrdamper') || isempty(config.data.mrdamper)
                config.data.mrdamper = struct();
            end
        case {"robotarmdata", "robotarm", "robot_arm", "robotarmdataset"}
            config.data.source_label = 'robotarmdata';
            if ~isfield(config.data, 'robotarm') || isempty(config.data.robotarm)
                config.data.robotarm = struct();
            end
            % MathWorks robotarmdata.mat uses Ts=5e-4 and the published
            % NLARX example downsamples by 10, then validates on uv3/yv3.
            config.data.robotarm = setDefaultField(config.data.robotarm, 'original_sampling_time', 5e-4);
            config.data.robotarm = setDefaultField(config.data.robotarm, 'downsample_factor', 10);
            config.data.robotarm = setDefaultField(config.data.robotarm, 'validation_experiment', 3);
        otherwise
            error('Unknown data source: %s', config.data.source);
    end

    if ~isfield(config, 'model') || ~isfield(config.model, 'tustin_sample_time') || isempty(config.model.tustin_sample_time)
        switch source
            case {"twotankdata", "twotank", "two_tank", "twotankdataset"}
                config.model.tustin_sample_time = config.data.twotank.sampling_time;
            case {"dryer2", "dryer", "dryerdataset"}
                config.model.tustin_sample_time = config.data.dryer2.sampling_time;
            otherwise
                config.model.tustin_sample_time = 1;
        end
    end
end

function s = setDefaultField(s, fieldName, value)
    if ~isfield(s, fieldName) || isempty(s.(fieldName))
        s.(fieldName) = value;
    end
end
