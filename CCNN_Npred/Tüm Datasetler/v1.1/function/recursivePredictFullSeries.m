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

    % Bu surumde diferansiyel aktivasyon sabit 1. derecedir; Tustin modunda ek state tutulur.
    z_history = cell(numel(W_local), 1);
    for h=1:numel(W_local)
        z_history{h, 1} = 0;
    end
    useTustinState = isfield(config, 'model') && isfield(config.model, 'activation') && contains(lower(string(config.model.activation)), "tustin");
    tustin_state = cell(numel(W_local), 1);
    if useTustinState
        for h=1:numel(W_local)
            tustin_state{h, 1} = 0;
        end
    end

    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; 
        
        for h=1:numel(W_local)
            z = x*W_local{h}; 
            
            % diff1: z - z_prev; Tustin modunda ek state kullanilir.
            if useTustinState
                [a, tustin_state{h, 1}] = applyHiddenActivation(z, z_history{h, 1}, g, config, tustin_state{h, 1});
            else
                a = applyHiddenActivation(z, z_history{h, 1}, g, config);
            end

            z_history{h, 1} = z;
            
            x=[x, a]; 
        end
        Yhat(k)= x * w_o_local;
    end
end
