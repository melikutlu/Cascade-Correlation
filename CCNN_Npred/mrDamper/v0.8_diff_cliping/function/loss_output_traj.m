function [L,grad] = loss_output_traj(w, X0, U, T, W_hidden, g, config)
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w, config);
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    L = l2loss(Yvec, Tvec, 'DataFormat', 'CB');  % MSE
    grad = dlgradient(L, w);
end
