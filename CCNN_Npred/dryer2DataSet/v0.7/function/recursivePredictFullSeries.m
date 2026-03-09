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
    orders = getHiddenOrders(config, numel(W_local));

    z_state = cell(numel(W_local),1);
    for h=1:numel(W_local)
        z_state{h} = struct('prev1', zeros(1,1), 'prev2', zeros(1,1));
    end

    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; for h=1:numel(W_local); z = x*W_local{h}; [a, z_state{h}] = applyHiddenActivation(z, z_state{h}, g, config, orders(h)); x=[x, a]; end
        Yhat(k)= x * w_o_local;
    end
end
