function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    maxLag = 0; if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    
    % +1 EKLEDİK (Kayıp yörüngeyi geri aldık)
    Ns = length(Y) - N - maxLag + 1; 
    
    if Ns<1; error('Not enough data'); end
    nu = numel(ulags); ny = numel(ylags);
    X0 = zeros(Ns, nu+ny); Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);
    
    for idx=1:Ns
        % -1 EKLEDİK (t=1'den, yani y1'den başlamasını sağladık)
        i = idx + maxLag - 1; 
        
        row = zeros(1,nu+ny);
        for j=1:nu; L=ulags(j); if L==0; row(j)=U(i); else row(j)=U(i+1-L); end; end
        for j=1:ny; L=ylags(j); row(nu+j)=Y(i+1-L); end
        X0(idx,:) = row; Useq(idx,:) = U(i+1:i+N)'; Tseq(idx,:) = Y(i+1:i+N)';
    end
end
