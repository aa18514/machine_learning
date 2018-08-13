function [error] = compute_error(y_predicted, y)
   erro = 0;
    for i = 1:length(y_predicted)
        if(y_predicted(i) ~= y(i))
            erro = erro + 1;
        end
    end
    error = erro/length(y_predicted);
end