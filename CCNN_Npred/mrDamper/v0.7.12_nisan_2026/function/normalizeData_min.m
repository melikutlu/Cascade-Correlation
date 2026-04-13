function [Utr,Ytr,Uva,Yva,stats] = normalizeData_min(method,Utr,Ytr,Uva,Yva)
    switch lower(method)
        case 'zscore'
            stats.u_mu = mean(Utr); 
            stats.u_std = std(Utr)+eps;
            stats.y_mu = mean(Ytr); 
            stats.y_std = std(Ytr)+eps;
            Utr = (Utr - stats.u_mu)/stats.u_std; 
            Uva = (Uva - stats.u_mu)/stats.u_std;
            Ytr = (Ytr - stats.y_mu)/stats.y_std; 
            Yva = (Yva - stats.y_mu)/stats.y_std;
        otherwise
            error('Unknown normalization');
    end
end
