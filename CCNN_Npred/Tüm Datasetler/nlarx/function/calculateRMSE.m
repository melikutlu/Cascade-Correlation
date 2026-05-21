function rmse = calculateRMSE(y_true, y_pred)
    % CALCULATERMSE Calculate Root Mean Squared Error
    % rmse = sqrt(mean((y_true - y_pred)^2))
    
    if length(y_true) ~= length(y_pred)
        error('y_true and y_pred must have the same length');
    end
    
    error = y_true - y_pred;
    mse = mean(error.^2);
    rmse = sqrt(mse);
end
