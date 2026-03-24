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
    % En son warmup adimindan y gecmisini initle
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

    % Warm-up phase: z history'yi gercek gecmisten doldur
    if nHidden > 0
        for w = 1:nWarmupSteps
            x_warmup = reshape(X0(:, w, 1:nu+ny), M, nu+ny);
            x_warmup = dlarray(x_warmup);
            
            if w > 1
                for j = 1:ny
                    yhist(:, ylags(j)) = X0(:, w, nu+j);
                end
            end
            
            x = x_warmup;
            for h = 1:nHidden
                z = x * W_hidden{h};
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
                z_history{h, 1} = z;
                x = [x, a];
            end
        end
    end

    % Main prediction phase
    for t = 1:N
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
                    uvals(:,j) = X0(:, nWarmupSteps, j);
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
            a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            z_history{h, 1} = z;
            x = [x, a];
        end
        
        y = x * w_o;
        Y(:,t) = y;
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end
