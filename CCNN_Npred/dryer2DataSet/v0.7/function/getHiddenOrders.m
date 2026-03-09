function orders = getHiddenOrders(config, nHidden)
    % Return per-hidden activation order list (1 or 2). Defaults to 1.
    if isfield(config, 'model') && isfield(config.model, 'hidden_orders') && ~isempty(config.model.hidden_orders)
        orders = config.model.hidden_orders(:)';
    else
        orders = ones(1, nHidden);
    end
    if numel(orders) < nHidden
        orders = [orders, ones(1, nHidden - numel(orders))];
    elseif numel(orders) > nHidden
        orders = orders(1:nHidden);
    end
end
