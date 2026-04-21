function outStr = formatArrayField(values)
    if isempty(values)
        outStr = '[]';
    else
        outStr = mat2str(values);
    end
end
