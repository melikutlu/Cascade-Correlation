function outStr = formatPlateauValue(val)
    if isempty(val) || isnan(val)
        outStr = 'none';
    else
        outStr = num2str(val);
    end
end
