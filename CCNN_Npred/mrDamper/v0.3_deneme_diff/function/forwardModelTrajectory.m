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
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    % Diferansiyel derecesini ve hidden katman sayısını al
    nHidden = numel(W_hidden);
    diffOrder = getDiffOrder(config);
    
    % Her hidden katman için geçmiş z değerlerini tut (diffOrder kadar)
    z_history = cell(nHidden, diffOrder + 1);  % +1 mevcut z için
    for h = 1:nHidden
        for d = 1:diffOrder + 1
            z_history{h, d} = dlarray(zeros(M, 1));
        end
    end

    % z geçmişini 0 ile değil, X0'dan türetilen bir önceki ağ durumu ile başlat
    if nHidden > 0 && diffOrder > 0
        x_prev = dlarray(X0(:, 1:nu+ny));
        for h = 1:nHidden
            z_prev_est = x_prev * W_hidden{h};
            z_history{h, 1} = z_prev_est;

            % Daha yüksek fark derecelerinde (örn. diff2) geçmiş bilinmediği için
            % en iyi eldeki tahmini geçmişe yayarak sıfır başlangıç etkisini azalt.
            for d = 2:diffOrder + 1
                z_history{h, d} = z_prev_est;
            end

            % Sonraki hidden katmanın z_prev tahmini için bir önceki katmanın
            % aktivasyonunu da üretip x_prev'e ekle.
            if diffOrder == 1
                a_prev = applyHiddenActivation(z_prev_est, z_prev_est, g, config);
            elseif diffOrder == 2
                a_prev = applyHiddenActivation(z_prev_est - 2*z_prev_est + z_prev_est, [], g, config);
            else
                a_prev = applyHiddenActivation(z_prev_est, [], g, config);
            end
            x_prev = [x_prev, a_prev];
        end
    end

    for t = 1:N
        % Regresör oluştur (aynı)
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
                    uvals(:,j) = X0(:, j);
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