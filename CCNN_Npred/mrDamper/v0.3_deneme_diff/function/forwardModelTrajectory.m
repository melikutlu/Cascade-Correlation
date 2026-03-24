function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); 
    N = size(U,2);
    Y = dlarray(zeros(M,N));

    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);

    % Bu surumde diferansiyel aktivasyon sabit 1. derecedir.
    nWarmupSteps = 2;
    
    % X0 shape: (M, nWarmupSteps, nu+ny)
    % En son warmup adımından (w=nWarmupSteps) y geçmişini initle
    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nWarmupSteps, nu+j);
    end

    % Her hidden katman icin bir onceki z degeri tutulur (diff1)
    nHidden = numel(W_hidden);
    z_history = cell(nHidden, 1);
    for h = 1:nHidden
        z_history{h, 1} = dlarray(zeros(M, 1));
    end

    % ========== WARM-UP PHASE ==========
    % X0'daki warm-up adımlarını ağdan geçir, z_history'yi gerçek verilerle popüle et
    % Bu aşamada hiçbir prediction output'ı kaydedilmez, sadece iç state initialized
    
    if nHidden > 0
        for w = 1:nWarmupSteps
            % Warm-up adımı w'den regresörleri al
            x_warmup = reshape(X0(:, w, 1:nu+ny), M, nu+ny);
            x_warmup = dlarray(x_warmup);
            
            % Y geçmişini warm-up'dan güncelle (eğer w > 1 ise)
            if w > 1
                for j = 1:ny
                    yhist(:, ylags(j)) = X0(:, w, nu+j);
                end
            end
            
            x = x_warmup;
            
            % Warm-up adımında hidden katmanları işle
            for h = 1:nHidden
                z = x * W_hidden{h};
                
                % diff1: z - z_prev
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);

                z_history{h, 1} = z;
                
                x = [x, a];
            end
        end
    end

    % ========== MAIN PREDICTION PHASE ==========
    % z_history artık gerçek geçmiş verileriyle popüle edilmiş
    for t = 1:N
        % U'dan geçerli zaman adımının regresörlerini oluştur
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
                    % t-L < 1: X0'dan en son warmup adımını kullan (w=nWarmupSteps)
                    uvals(:,j) = X0(:, nWarmupSteps, j);
                end
            end
        end
        
        % Y geçmişinden y regresörleri al
        yvals = zeros(M, ny);
        for j = 1:ny
            yvals(:,j) = yhist(:, ylags(j));
        end

        x = dlarray([uvals, yvals]);
        
        for h = 1:nHidden
            z = x * W_hidden{h};
            
            % diff1: z - z_prev
            a = applyHiddenActivation(z, z_history{h, 1}, g, config);

            z_history{h, 1} = z;
            
            x = [x, a];
        end
        
        y = x * w_o;
        Y(:,t) = y;
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end