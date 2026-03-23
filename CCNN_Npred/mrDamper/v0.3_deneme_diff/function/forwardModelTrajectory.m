function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); 
    N = size(U,2);
    Y = dlarray(zeros(M,N));

    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);

    % Diferansiyel derecesini al
    diffOrder = getDiffOrder(config);
    nWarmupSteps = diffOrder + 1;
    
    % X0 shape: (M, nWarmupSteps, nu+ny)
    % En son warmup adımından (w=nWarmupSteps) y geçmişini initle
    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nWarmupSteps, nu+j);
    end

    % Her hidden katman için geçmiş z değerlerini tut (diffOrder+1 adet)
    nHidden = numel(W_hidden);
    z_history = cell(nHidden, diffOrder + 1);
    for h = 1:nHidden
        for d = 1:diffOrder + 1
            z_history{h, d} = dlarray(zeros(M, 1));
        end
    end

    % ========== WARM-UP PHASE ==========
    % X0'daki warm-up adımlarını ağdan geçir, z_history'yi gerçek verilerle popüle et
    % Bu aşamada hiçbir prediction output'ı kaydedilmez, sadece iç state initialized
    
    if nHidden > 0 && diffOrder > 0
        for w = 1:nWarmupSteps
            % Warm-up adımı w'den regresörleri al
            x_warmup = dlarray(X0(:, w, 1:nu+ny));
            
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
                
                % Diferansiyel hesapla
                if diffOrder == 1
                    % z - z_prev
                    a = applyHiddenActivation(z, z_history{h, 1}, g, config);
                elseif diffOrder == 2
                    % z - 2*z_prev + z_prev2
                    z_prev = z_history{h, 1};
                    z_prev2 = z_history{h, 2};
                    diff_z = z - 2*z_prev + z_prev2;
                    a = applyHiddenActivation(diff_z, [], g, config);
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
            
            % Diferansiyel hesapla
            if diffOrder == 1
                % 1. derece fark: z - z_prev
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            elseif diffOrder == 2
                % 2. derece fark: z - 2*z_prev + z_prev2
                z_prev = z_history{h, 1};
                z_prev2 = z_history{h, 2};
                diff_z = z - 2*z_prev + z_prev2;
                a = applyHiddenActivation(diff_z, [], g, config);
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