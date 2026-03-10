function [a, state] = applyHiddenActivation(z, state, g, config, order)
    % Apply activation with optional finite-difference order (0,1,2).
    % order=0: g(z); order=1: g(diff1); order=2: g(diff2).
    if nargin < 2 || isempty(state)
        state = struct('prev1', zeros(size(z), 'like', z), 'prev2', zeros(size(z), 'like', z));
    end
    if nargin < 5 || isempty(order)
        order = 1;
    end

    % compute finite differences
    diff1 = z - state.prev1;
    diff2 = diff1 - (state.prev1 - state.prev2);
    state.prev2 = state.prev1;
    state.prev1 = z;

    if order <= 0
        base = z;      % no diff
    elseif order == 1
        base = diff1;  % first diff
    else
        base = diff2;  % second diff
    end

    a = g(base);
end
