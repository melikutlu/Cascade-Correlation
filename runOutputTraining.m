% runOutputTraining.m
function [w_trained, E_residual, current_mse] = runOutputTraining(method, X, T, w_initial, max_epochs, params)
% method: config.output_trainer değeri (string)
% params: Tüm hiperparametreleri (eta, mu, batch_size vb.) içeren bir struct
    
    switch method
        case 'Quickprop_DL'
            [w_trained, E_residual, current_mse] = trainOutputLayer_Quickprop_With_dlgrad(X, T, w_initial, ...
                                                      max_epochs, params.eta_output, params.mu, params.epsilon);
        case 'GD_Autograd'
            [w_trained, E_residual, current_mse] = trainOutputLayer_GD_Autograd(X, T, w_initial, ...
                                                      max_epochs, params.eta_output, params.batch_size);
        case 'GD_Fullbatch'
            [w_trained, E_residual, current_mse] = trainOutputLayer_GD_fullbatch(X, T, w_initial, ...
                                                      max_epochs, params.eta_output_gd);
        case 'GD_MiniBatch'
            [w_trained, E_residual, current_mse] = trainOutputLayer_GD(X, T, w_initial, ...
                                                      max_epochs, params.eta_output_gd, params.batch_size);
        case 'Quickprop_Org'
            [w_trained, E_residual, current_mse] = trainOutputLayer(X, T, w_initial, ...
                                                      max_epochs, params.eta_output, params.mu, params.epsilon);
        otherwise
            error('Geçersiz eğitim metodu seçimi: %s', method);
    end
end