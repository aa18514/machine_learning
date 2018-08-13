%(sign((w0*X).*Y) is a vector with values +1 to indicate correctly
%classified or -1 to indicate incorrectly classified.Considering we have N
%points, if all these points are correctly classified we have N as sum of
%correctly classified points. This minus the sum of correctly and
%incorrectly classified points is 2 times the sum of incorrectly classified
%points. 
function [d] = compute_error(y,w0, X, N)
c = (sign((w0*X).*y));
m = (size(y, 2) - sum(c))/2;
d = m;
end