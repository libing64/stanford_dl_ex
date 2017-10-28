function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%

% for i = 1 : m
%   t = dot(theta, X(:, i));
%   f = f - ( y(i) * log(sigmoid(t)) +  (1 - y(i)) * (1 - log(sigmoid(t)) ) );
%   g = g + X(:, i) * (sigmoid(t) - y(i));
% end

y_hat = theta' * X; 
f = -(y * log(sigmoid(y_hat))' + (1 - y) * (1 - log(sigmoid(1 - y_hat)))');
g = X * (sigmoid(y_hat) - y)';

