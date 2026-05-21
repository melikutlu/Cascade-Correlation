function [Utr, Ytr, Uva, Yva] = loadDataByConfig_min(config)
    config = applyDatasetDefaults(config);
    source = lower(char(config.data.source_label));

    switch source
        case 'twotankdata'
            load twotankdata.mat; % must contain u,y
            u = u(:); y = y(:);
            w = config.data.twotank.warmup_samples;
            u = u(w+1:end); y = y(w+1:end);
            % optional filter
            if isfield(config.data.twotank, 'filter_cutoff') && config.data.twotank.filter_cutoff > 0
                fc = config.data.twotank.filter_cutoff; Ts = config.data.twotank.sampling_time;
                a = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts);
                uf = zeros(size(u)); yf = zeros(size(y)); uf(1)=u(1); yf(1)=y(1);
                for k=2:length(u)
                    uf(k) = a*u(k) + (1-a)*uf(k-1);
                    yf(k) = a*y(k) + (1-a)*yf(k-1);
                end
                u = uf; y = yf;
            end
            N = length(u); Ntr = floor(config.data.train_ratio * N);
            Utr = u(1:Ntr); Ytr = y(1:Ntr); Uva = u(Ntr+1:end); Yva = y(Ntr+1:end);
        case 'dryer2'
            load dryer2; % contains u2, y2
            Ts = config.data.dryer2.sampling_time;
            z_full = iddata(y2(:), u2(:), Ts);
            N_total = length(z_full.y);
            train_end = floor(config.data.train_ratio * N_total);
            val_end   = train_end + floor(config.data.val_ratio * N_total);
            z1 = z_full(1:train_end);
            z1f = detrend(z1);
            z2 = z_full(train_end+1:val_end);
            z2f = detrend(z2);
            Utr = z1f.u; Ytr = z1f.y;
            Uva = z2f.u; Yva = z2f.y;
        case 'mrdamper'
            load mrdamper.mat   % dataset variables: V, F
            u = V(:);   % velocity input
            y = F(:);   % force output
            N = length(u);
            Ntr = floor(config.data.train_ratio * N);
            Utr = u(1:Ntr);
            Ytr = y(1:Ntr);
            Uva = u(Ntr+1:end);
            Yva = y(Ntr+1:end);
        case 'robotarmdata'
            load robotarmdata.mat; % variables: ue, ye, uv1, yv1, uv2, yv2, uv3, yv3

            valExp = config.data.robotarm.validation_experiment;
            if ischar(valExp) || isstring(valExp)
                valExp = str2double(valExp);
            end
            if ~isscalar(valExp) || isnan(valExp) || ~ismember(valExp, 1:3)
                error('config.data.robotarm.validation_experiment must be 1, 2, or 3.');
            end

            valInputs = {uv1, uv2, uv3};
            valOutputs = {yv1, yv2, yv3};

            Ts = config.data.robotarm.original_sampling_time;
            downsampleFactor = config.data.robotarm.downsample_factor;
            if ischar(downsampleFactor) || isstring(downsampleFactor)
                downsampleFactor = str2double(downsampleFactor);
            end
            if ~isscalar(downsampleFactor) || isnan(downsampleFactor) || downsampleFactor < 1 || downsampleFactor ~= round(downsampleFactor)
                error('config.data.robotarm.downsample_factor must be a positive integer.');
            end

            eData = iddata(ye(:), ue(:), Ts, ...
                'InputName', 'Torque', ...
                'OutputName', 'Angular Velocity', ...
                'Tstart', 0);
            vData = iddata(valOutputs{valExp}(:), valInputs{valExp}(:), Ts, ...
                'InputName', 'Torque', ...
                'OutputName', 'Angular Velocity', ...
                'Tstart', 0);

            eData = idresamp(eData, [1 downsampleFactor]);
            vData = idresamp(vData, [1 downsampleFactor]);
            eData.Name = 'estimation data';
            vData.Name = 'validation data';

            Utr = eData.u;
            Ytr = eData.y;
            Uva = vData.u;
            Yva = vData.y;
        otherwise
            error('Unknown data source: %s', config.data.source);
    end
end
