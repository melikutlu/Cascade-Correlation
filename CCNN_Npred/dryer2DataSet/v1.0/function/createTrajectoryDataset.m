function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    maxLag = 0; 
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end

    % Bu surumde diferansiyel aktivasyon sabit 1. derecedir.
    diffOrder = 1;
    nWarmupSteps = 2; % diff1 icin iki adimlik gecmis gerekir

    % Baslangic gecikmesini maxHidden ile bagla:
    % totalInitDelay = maxLag + (maxHidden * diffOrder)
    if isfield(config, 'model') && isfield(config.model, 'max_hidden_units') && ~isempty(config.model.max_hidden_units)
        nHiddenMax = max(config.model.max_hidden_units(:));
    else
        nHiddenMax = 0;
    end
    totalInitDelay = maxLag + (nHiddenMax * diffOrder);

    % Veri sayisi
    Ns = length(Y) - N - totalInitDelay + 1; 

    if Ns < 1
        error('Not enough data. Need at least %d samples for %d-step predictions with diff1 and maxHidden=%d', ...
              N + totalInitDelay, N, nHiddenMax);
    end

    nu = numel(ulags); 
    ny = numel(ylags);

    % X0 yapi: (Ns, nWarmupSteps, nu+ny)
    X0 = zeros(Ns, nWarmupSteps, nu+ny); 
    Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);

    for idx = 1:Ns
        % Baslangic indeksi
        i = idx + totalInitDelay - 1;

        % Warm-up adimlarini doldur (en eskiden en yeniye)
        for w = 1:nWarmupSteps
            time_idx = i - nWarmupSteps + w;

            row = zeros(1, nu+ny);
            for j = 1:nu
                L = ulags(j);
                if L == 0
                    row(j) = U(time_idx);
                else
                    row(j) = U(time_idx + 1 - L);
                end
            end
            for j = 1:ny
                L = ylags(j);
                row(nu+j) = Y(time_idx + 1 - L);
            end
            X0(idx, w, :) = row;
        end

        % Tahmin sekansi
        Useq(idx,:) = U(i+1:i+N)'; 
        Tseq(idx,:) = Y(i+1:i+N)';
    end
end
