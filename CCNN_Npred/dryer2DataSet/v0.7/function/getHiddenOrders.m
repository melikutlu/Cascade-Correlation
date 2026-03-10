function orders = getHiddenOrders(config, nHidden)
    % Return per-hidden activation order list (0,1,2). Defaults to 1 (no diff provided → 1st diff).
    if isfield(config, 'model') && isfield(config.model, 'hidden_orders') && ~isempty(config.model.hidden_orders)
        orders = config.model.hidden_orders(:)';
    elseif isfield(config, 'model') && isfield(config.model, 'order') && ~isempty(config.model.order)
        orders = repmat(config.model.order, 1, nHidden);
    else
        orders = ones(1, nHidden);
    end
    if numel(orders) < nHidden
        orders = [orders, repmat(orders(end), 1, nHidden - numel(orders))];
    elseif numel(orders) > nHidden
        orders = orders(1:nHidden);
    end
end
