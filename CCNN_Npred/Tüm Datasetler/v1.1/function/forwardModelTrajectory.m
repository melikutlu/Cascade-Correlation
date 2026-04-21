function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); 
    N = size(U,2);
    Y = dlarray(zeros(M,N));

    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);

    % Turev operatoru aktivasyonda hesaplanir; ilk adimda z_prev = 0 alinir.
    nWarmupSteps = 1;
    
    % X0 shape: (M, nWarmupSteps, nu+ny)
    % Bu surumde nWarmupSteps=1 oldugu icin t0 gecmisinden basla.
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

    % ========== MAIN PREDICTION PHASE ==========
    % z_history her hidden icin sifirdan baslar; warm-up precompute yoktur.
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
                    % t-L < 1: X0'daki tek warm-up adimini kullan
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
    
    a = applyHiddenActivation(z, z_history{h,1}, g, config);

    
    if t <= 20  % sadece ilk 5 adımı logla
    end
    z_history{h,1} = z;
    x = [x, a];
        end
        
        y = x * w_o;
        Y(:,t) = y;
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end