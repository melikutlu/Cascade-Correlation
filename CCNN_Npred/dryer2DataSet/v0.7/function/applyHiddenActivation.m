function [a, state] = applyHiddenActivation(z, state, g, config, order)
    % Apply activation with time-difference transforms; supports first or second order.
    if nargin < 2 || isempty(state)
        state = struct('prev1', zeros(size(z), 'like', z), 'prev2', zeros(size(z), 'like', z));
    end
    if nargin < 5 || isempty(order)
        order = 1;
    end
    mode = 'tanh';
    if isfield(config, 'model') && isfield(config.model, 'activation') && ~isempty(config.model.activation)
        mode = lower(string(config.model.activation));
    end

    % compute finite differences
    diff1 = z - state.prev1;
    diff2 = diff1 - (state.prev1 - state.prev2);
    state.prev2 = state.prev1;
    state.prev1 = z;

    switch mode
        case {"diff", "diff-only", "diff-tanh", "diff_tanh"}
            if order == 2
                base = diff2;
            else
                base = diff1;
            end
            if contains(mode, 'tanh')
                a = g(base);
            else
                a = base;
            end
        otherwise
            a = g(z);
    end
end
