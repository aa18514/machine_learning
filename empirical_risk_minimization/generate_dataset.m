%generate data_set of cardinality N 
%introduce noise, comes from the probability distribution
%noise is a vector of cardinality N, if the ith value exceeds 0.1, give
%label + 1, otherwise give label -1
%pointwise multiplication with the y vector
function[y] = generate_dataset(X_1, N)
y = zeros(1, N);
for i = 1:N
    y(i) = 2*double(X_1(2, i) >= X_1(1, i)*(X_1(1, i) - 1)*(X_1(1, i) - 2)) - 1; 
end
noise = 2*(rand(1,N) >= 0.1) -1;
y = y.*noise;
end