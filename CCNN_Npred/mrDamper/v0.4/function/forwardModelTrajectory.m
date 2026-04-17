function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); 
    N = size(U,2);
    Y = dlarray(zeros(M,N));

    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);

    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    
    % Diferansiyel derecesini ve hidden katman sayısını al
    nHidden = numel(W_hidden);
    diffOrder = getDiffOrder(config);
    
    % X0'ın yapısını anla: warmupSteps = diffOrder + 1
    warmupSteps = diffOrder + 1;
    nRegressor = nu + ny;
    
    % X0'ın boyutunun doğru olduğunu kontrol et
    if size(X0, 2) ~= warmupSteps * nRegressor
        error('X0 column size (%d) must equal warmupSteps (%d) * nRegressor (%d)', ...
              size(X0,2), warmupSteps, nRegressor);
    end
    
    % Her hidden katman için geçmiş z değerlerini tut (diffOrder kadar)
    z_history = cell(nHidden, diffOrder + 1);  % +1 mevcut z için
    for h = 1:nHidden
        for d = 1:diffOrder + 1
            z_history{h, d} = dlarray(zeros(M, 1));
        end
    end

    % ========== WARM-UP PHASE ==========
    % X0'daki warmupSteps adet zaman adımını sırayla ağa besle
    % Bu, z_history'yi gerçek geçmiş verilerle doldurur
    
    for step = 1:warmupSteps
        % X0'ın step'inci satırını çıkar
        col_start = (step-1)*nRegressor + 1;
        col_end = step*nRegressor;
        x_warmup = dlarray(X0(:, col_start:col_end));
        
        uvals_warmup = x_warmup(:, 1:nu);
        yvals_warmup = x_warmup(:, nu+1:nu+ny);
        
        % Y history'yi güncelle (warm-up'ta sadece input y'den)
        for j = 1:ny
            yhist(:, ylags(j)) = yvals_warmup(:, j);
        end
        
        % Regressörleri oluştur (warm-up'ta basit: gelen veri direkt kullanılır)
        x_combined = dlarray(x_warmup);
        
        % Hidden katmanları hesapla
        for h = 1:nHidden
            z = x_combined * W_hidden{h};
            
            % Diferansiyel hesapla
            if diffOrder == 1
                % 1. derece fark: z - z_prev
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            elseif diffOrder == 2
                % 2. derece fark: z - 2*z_prev + z_prev2
                z_prev = z_history{h, 1};
                z_prev2 = z_history{h, 2};
                diff_z = z - 2*z_prev + z_prev2;
                a = applyHiddenActivation(diff_z, [], g, config);  % z_prev'i boş geç
            else
                a = applyHiddenActivation(z, [], g, config);
            end
            
            % Geçmişi güncelle (kaydır)
            for d = diffOrder:-1:1
                z_history{h, d+1} = z_history{h, d};
            end
            z_history{h, 1} = z;
            
            x_combined = [x_combined, a];
        end
    end
    
    % ========== MAIN PREDICTION PHASE ==========
    % Şimdi z_history doğru şekilde dolduruldu, N adımlık tahmin başlayabilir
    
    for t = 1:N
        % Regresör oluştur
        uvals = zeros(M, nu);
        for j = 1:nu
            L = ulags(j);
            if L == 0
                uvals(:,j) = U(:, t);
            else
                idx = t - L;
                if idx >= 1
                    uvals(:,j) = U(:, idx);
                else
                    % Eğer negatif indeks, warm-up'ın ötesinde gitmemeli
                    % Bu normal tahmin döngüsünde olmamalı ama güvenlik için sıfır ata
                    uvals(:,j) = 0;
                end
            end
        end
        
        yvals = zeros(M, ny);
        for j = 1:ny
            yvals(:,j) = yhist(:, ylags(j));
        end

        x = dlarray([uvals, yvals]);
        
        for h = 1:nHidden
            z = x * W_hidden{h};
            
            % Diferansiyel hesapla
            if diffOrder == 1
                % 1. derece fark: z - z_prev
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            elseif diffOrder == 2
                % 2. derece fark: z - 2*z_prev + z_prev2
                z_prev = z_history{h, 1};
                z_prev2 = z_history{h, 2};
                diff_z = z - 2*z_prev + z_prev2;
                a = applyHiddenActivation(diff_z, [], g, config);  % z_prev'i boş geç
            else
                a = applyHiddenActivation(z, [], g, config);
            end
            
            % Geçmişi güncelle (kaydır)
            for d = diffOrder:-1:1
                z_history{h, d+1} = z_history{h, d};
            end
            z_history{h, 1} = z;
            
            x = [x, a];
        end
        
        y = x * w_o;
        Y(:,t) = y;
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end