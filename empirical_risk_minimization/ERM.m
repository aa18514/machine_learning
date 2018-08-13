clc
clear
close all

N = [10, 100, 10000];
pF1 = @(x1, x2) [x2]; 
pF2 = @(x1, x2) [x1; x2]; 
pF3 = @(x1, x2) [x1.^2; x1; x2]; 
pF4 = @(x1, x2) [x1.^3; x1.^2; x1; x2];
pF5 = @(x1, x2) [x1.^4 ;x1.^3 ;x1.^2; x1; x2];
compute_complexity = @(dvcH, n, w, delta) (( ( ((8*dvcH)/n) * log((2*n*exp(1))/dvcH) ) + ( 8/n * log(4/(w*delta)) ))^0.5);
ERM_training_error = zeros(3, 5);
ERM_test_error = zeros(3, 5);
SRM_test_error_c0 = zeros(3, 5);
SRM_test_error_c1  = zeros(3, 5);
SRM_test_error_c2 = zeros(3, 5);
complexity = zeros(3, 5);
test_dataset_size = 10000;
dvcH = [1, 2, 3, 4, 5];
for j = 1:3
    for i = 1:5
        complexity(j, i) = compute_complexity(dvcH(i), N(j), 0.2, 0.1);
    end
end
for j = 1:3
    X_train = [2*rand(1, N(j)) - 1; 2*rand(1, N(j)) - 1];
    X_test =  [3*rand(1, test_dataset_size) - 2; 3*rand(1, test_dataset_size) - 2]; 
    [y_train] = generate_dataset(X_train, N(j));
    [y_test]  = generate_dataset(X_test, test_dataset_size);
    for i = 1:100
        [w0_optimal, best_training_error_0] = pocket_learning_algorithm(pF1(X_train(1,:), X_train(2, :)), y_train, [0], N(j));
        ERM_training_error(j, 1) = ERM_training_error(j, 1) +  best_training_error_0; 
        
        [w1_optimal, best_training_error_1] = pocket_learning_algorithm(pF2(X_train(1, :), X_train(2, :)),  y_train, [0, 0], N(j)); 
        ERM_training_error(j, 2) = ERM_training_error(j, 2) +  best_training_error_1;
        
        [w2_optimal, best_training_error_2] = pocket_learning_algorithm(pF3(X_train(1, :), X_train(2, :)), y_train, [0, 0, 0], N(j));  
        ERM_training_error(j, 3) = ERM_training_error(j, 3) + best_training_error_2;
        
        [w3_optimal, best_training_error_3] = pocket_learning_algorithm(pF4(X_train(1, :), X_train(2, :)), y_train, [0, 0, 0, 0], N(j)); 
        ERM_training_error(j, 4) = ERM_training_error(j, 4) + best_training_error_3;
        
        [w4_optimal, best_training_error_4] = pocket_learning_algorithm(pF5(X_train(1, :), X_train(2, :)), y_train, [0, 0, 0, 0, 0], N(j));  
        ERM_training_error(j, 5) = ERM_training_error(j, 5) + best_training_error_4;
    end
    
     y_test_error_1 = compute_error(y_test, w0_optimal, pF1(X_test(1,:), X_test(2, :)), test_dataset_size)/test_dataset_size;
     y_test_error_2 = compute_error(y_test, w1_optimal, pF2(X_test(1,:), X_test(2, :)), test_dataset_size)/test_dataset_size;
     y_test_error_3 = compute_error(y_test, w2_optimal, pF3(X_test(1,:), X_test(2, :)), test_dataset_size)/test_dataset_size;
     y_test_error_4 = compute_error(y_test, w3_optimal, pF4(X_test(1,:), X_test(2, :)), test_dataset_size)/test_dataset_size;
     y_test_error_5 = compute_error(y_test, w4_optimal, pF5(X_test(1,:), X_test(2, :)), test_dataset_size)/test_dataset_size;
     
     
     ERM_training_error(j,:) = ERM_training_error(j,:)/100; 
     ERM_test_error(j,:) = [y_test_error_1, y_test_error_2, y_test_error_3, y_test_error_4, y_test_error_5]; 
     SRM_test_error_c0(j,:) = ERM_training_error(j, :) + complexity(j, :);
     SRM_test_error_c1(j,:) = ERM_training_error(j, :) + 0.1*complexity(j, :);
     SRM_test_error_c2(j,:) = ERM_training_error(j, :) + 0.01*complexity(j, :);
end