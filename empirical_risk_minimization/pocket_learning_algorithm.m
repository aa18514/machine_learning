function[d, a] = pocket_learning_algorithm(x, y, w, N)
k = 1;
j = 1; 
w_optimal = w;
best_training_error = compute_error(y, w, x, N);
while (j < N + 1) && (k < 20000)
    if((w*x(j))*y(j) <= 0)
        w = w + (x(j)*y(j));
        training_error = compute_error(y, w, x, N);
        if(training_error < best_training_error)
            w_optimal = w; 
            best_training_error = training_error;
        end 
        k = k + 1;
        j = 1; 
    else 
        j = j + 1;
    end
end
d = w_optimal;
a = best_training_error/N;
end
