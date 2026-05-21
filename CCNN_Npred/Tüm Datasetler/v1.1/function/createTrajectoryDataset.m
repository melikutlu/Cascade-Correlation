function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    maxLag = 0; 
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    
    % Turev operatoru hidden aktivasyonunda uygulanir; veri setinde ek warm-up
    % gecikmesi kullanilmaz. Ilk adimda z_prev=0 kabul edilir.
    nWarmupSteps = 1;
    totalInitDelay = maxLag;
    
    % Veri sayısı
    Ns = length(Y) - N - totalInitDelay + 1; 


    
    if Ns < 1
        error('Not enough data. Need at least %d samples for %d-step predictions', ...
              N + totalInitDelay);
    end
    
    nu = numel(ulags); 
    ny = numel(ylags);
    
    % X0 yapi: (Ns, nWarmupSteps, nu+ny)
    % Bu surumde nWarmupSteps=1 ve sadece t0 regresorleri saklanir.
    X0 = zeros(Ns, nWarmupSteps, nu+ny); 
    Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);
    
    for idx = 1:Ns
        % Başlangıç indeksi
        i = idx + totalInitDelay - 1;
        
        % Tek warm-up adimi (t0 regresorleri)
        for w = 1:nWarmupSteps
            % Gercek zaman indeksi
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
        
        % Tahmin sekansı
        Useq(idx,:) = U(i+1:i+N)'; 
        Tseq(idx,:) = Y(i+1:i+N)';
    end
end