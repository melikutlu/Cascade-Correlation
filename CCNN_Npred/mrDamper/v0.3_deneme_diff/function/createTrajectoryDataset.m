function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    maxLag = 0; 
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    
    % Diferansiyel derecesini al (activation tipine bağlı)
    diffOrder = getDiffOrder(config);
    
    % Warm-up için gereken zaman adımları: diffOrder kadar
    % Her diff derecesi için: diff1->2 adım, diff2->3 adım, etc.
    nWarmupSteps = diffOrder + 1;
    
    % Toplam başlangıç gecikmesi = maxLag + (nWarmupSteps - 1)
    % (nWarmupSteps - 1) çünkü warm-up için son nWarmupSteps zaman adımının regresörlerini ihtiyaç duyarız
    totalInitDelay = maxLag + (nWarmupSteps - 1);
    
    % Veri sayısı
    Ns = length(Y) - N - totalInitDelay + 1; 
    
    if Ns < 1
        error('Not enough data. Need at least %d samples for %d-step predictions with diffOrder=%d (nWarmupSteps=%d)', ...
              N + totalInitDelay, N, diffOrder, nWarmupSteps);
    end
    
    nu = numel(ulags); 
    ny = numel(ylags);
    
    % X0 yapı: (Ns, nWarmupSteps, nu+ny)
    % Her trajectory için son nWarmupSteps zaman adımının regresörlerini sakla
    X0 = zeros(Ns, nWarmupSteps, nu+ny); 
    Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);
    
    for idx = 1:Ns
        % Başlangıç indeksi
        i = idx + totalInitDelay - 1;
        
        % Warm-up adımlarını doldur (geçmiş zaman adımlarından en yeni olana doğru)
        for w = 1:nWarmupSteps
            % w=1 en eski, w=nWarmupSteps en yeni (t=0)
            % Gerçek zaman indeksi: i - (nWarmupSteps - w) = i - nWarmupSteps + w
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