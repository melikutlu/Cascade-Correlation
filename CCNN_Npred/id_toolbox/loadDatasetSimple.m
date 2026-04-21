function [Utr, Ytr, Uva, Yva, Ts, dataInfo] = loadDatasetSimple(config)

    switch lower(config.data.source)

        case 'twotankdata'
            load twotankdata.mat;   % must contain u, y
            u = u(:);
            y = y(:);

            Ts = config.data.twotank.sampling_time;

            % warmup remove
            w = config.data.twotank.warmup_samples;
            if w > 0
                u = u(w+1:end);
                y = y(w+1:end);
            end

            % optional simple low-pass filter
            if isfield(config.data.twotank, 'filter_cutoff') && config.data.twotank.filter_cutoff > 0
                fc = config.data.twotank.filter_cutoff;
                a = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts);

                uf = zeros(size(u));
                yf = zeros(size(y));
                uf(1) = u(1);
                yf(1) = y(1);

                for k = 2:length(u)
                    uf(k) = a*u(k) + (1-a)*uf(k-1);
                    yf(k) = a*y(k) + (1-a)*yf(k-1);
                end

                u = uf;
                y = yf;
            end

        case 'dryer2'
            load dryer2;   % usually contains u2, y2
            u = u2(:);
            y = y2(:);
            Ts = config.data.dryer2.sampling_time;

            % optional detrend like your earlier code
            z = iddata(y, u, Ts);
            z = detrend(z);
            u = z.u;
            y = z.y;

        case 'mrdamper'
            load mrdamper.mat;  % expected V, F

            % adjust here if your variable names differ
            if exist('V', 'var') && exist('F', 'var')
                u = V(:);
                y = F(:);
            else
                error(['mrdamper.mat loaded but expected variables V and F were not found. ', ...
                       'Please adapt loadDatasetSimple.m for your local variable names.']);
            end

            Ts = config.data.mrdamper.sampling_time;

        otherwise
            error('Unknown data source: %s', config.data.source);
    end

    % basic checks
    if length(u) ~= length(y)
        error('Input and output lengths do not match.');
    end

    N = length(u);
    Ntr = floor(config.data.train_ratio * N);

    if Ntr < 2 || Ntr >= N
        error('Invalid training ratio. Training set size became invalid.');
    end

    Utr = u(1:Ntr);
    Ytr = y(1:Ntr);
    Uva = u(Ntr+1:end);
    Yva = y(Ntr+1:end);

    dataInfo = struct();
    dataInfo.name = config.data.source;
    dataInfo.Ntotal = N;
    dataInfo.Ntrain = length(Utr);
    dataInfo.Nval   = length(Uva);
    dataInfo.Ts     = Ts;
end