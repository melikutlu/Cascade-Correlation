function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    maxLag = 0; 
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    
    % Hidden katman sayısı ve diferansiyel derecesini hesapla
    nHidden = numel(config.model.max_hidden_units);  % Hidden katman sayısı
    diffOrder = getDiffOrder(config);  % Aktivasyon tipine göre diferansiyel derecesi
    
    % Toplam başlangıç gecikmesi = maxLag + (nHidden × diffOrder)
    totalInitDelay = maxLag + (nHidden * diffOrder);
    
    % +1 ekstra başlangıç için
    Ns = length(Y) - N - totalInitDelay + 1; 
    
    if Ns < 1
        error('Not enough data. Need at least %d samples for %d-step predictions with %d hidden layers (diffOrder=%d)', ...
              N + totalInitDelay, N, nHidden, diffOrder);
    end
    
    nu = numel(ulags); 
    ny = numel(ylags);
    X0 = zeros(Ns, nu+ny); 
    Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);
    
    for idx = 1:Ns
        % Başlangıç indeksi: totalInitDelay kadar kaydır
        i = idx + totalInitDelay - 1; 
        
        row = zeros(1, nu+ny);
        for j = 1:nu
            L = ulags(j); 
            if L == 0
                row(j) = U(i);
            else
                row(j) = U(i+1-L);
            end
        end
        for j = 1:ny
            L = ylags(j); 
            row(nu+j) = Y(i+1-L);
        end
        X0(idx,:) = row; 
        Useq(idx,:) = U(i+1:i+N)'; 
        Tseq(idx,:) = Y(i+1:i+N)';
    end
end