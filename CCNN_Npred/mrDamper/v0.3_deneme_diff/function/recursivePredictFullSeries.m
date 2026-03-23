function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); 
    Yhat = zeros(N,1); 
    if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)'; 
    nu=numel(ulags); ny=numel(ylags);

    % work with gathered weights for speed during inference
    W_local = cellfun(@gather, W_hidden, 'UniformOutput', false);
    w_o_local = gather(w_o);

    % Diferansiyel derecesi
    diffOrder = getDiffOrder(config);
    
    % Her hidden katman için z geçmişi (diffOrder+1 adım)
    z_history = cell(numel(W_local), diffOrder + 1);
    for h=1:numel(W_local)
        for d=1:diffOrder + 1
            z_history{h, d} = 0;
        end
    end

    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; 
        
        for h=1:numel(W_local)
            z = x*W_local{h}; 
            
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
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            end
            
            % Geçmişi güncelle (kaydır)
            for d = diffOrder:-1:1
                z_history{h, d+1} = z_history{h, d};
            end
            z_history{h, 1} = z;
            
            x=[x, a]; 
        end
        Yhat(k)= x * w_o_local;
    end
end
