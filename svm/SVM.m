clc
clear

[y_train, mod_feature_train] = extract_features_and_labels('zip.train', 257); 
[y_test, mod_feature_test] = extract_features_and_labels('zip.test', 257); 
rng(1); 

%columns in error represent the effect of changing the value of C%
%rows in error represent the effect of changing the value of gamma%
error = zeros(11, 6); 

SVMModel = fitcsvm(mod_feature_train, y_train, 'KernelFunction', 'gaussian', 'KernelScale', 'auto'); 
ks = SVMModel.KernelParameters.Scale; 
gamma = [(1e-8), (1e-5), (1e-4), (1e-3), (1e-2), (1e-1), (1e+0), (1e+1), (1e+2), (1e+3),(1e+4)];
C_values = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000];

for i = 1:length(gamma)
    for j = 1:length(C_values)
        SVMModel = fitcsvm(mod_feature_train, y_train, 'KernelFunction', 'gaussian', 'KernelScale', gamma(i), 'BoxConstraint', C_values(j));
        CSVMModel = crossval(SVMModel); 
        error(i, j) = kfoldLoss(CSVMModel)*100; 
    end
end

for j = 1:length(C_values) 
    figure(j)
    error_gamma = error(:,j); %j-th column, all corresponding rows
    plot(gamma, error_gamma);
    str= sprintf('plot with different C value %f', C_values(j));
    title(str);
    xlabel('gamma(log)'); 
    ylabel('Kfold % error');
    set(gca, 'XScale', 'log');
end

best_gamma = 0; 
best_C = 0;
min_error = 1000;
for i = 1:length(gamma)
    for j = 1:length(C_values)
        if((error(i, j) < min_error))
            min_error = error(i, j);
            best_gamma = gamma(i); 
            best_C = C_values(j); 
        end
    end
end

SVMModel = fitcsvm(mod_feature_train, y_train, 'KernelFunction', 'gaussian', 'KernelScale', best_gamma, 'BoxConstraint', best_C);
label = predict(SVMModel, mod_feature_test); 
[test_error] = compute_error(label, y_test); 
label = predict(SVMModel, mod_feature_train); 
[train_error] = compute_error(label, y_train);