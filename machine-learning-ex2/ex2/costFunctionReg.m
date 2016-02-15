function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(theta' * X');

h_prime = h'; % in use twice, therefore convenient to cache

J = (-1 / m) * sum(y .* log(h_prime) + (1 - y) .* log(1 - h_prime));

grad = (1 / m) * (h - y') * X;

% Regularize theta(2:n)
n = length(theta);

J = J + (lambda / (2 * m)) * sum((theta(2:n) .^ 2));

grad(2:n) = grad(2:n) + (lambda / m) * (theta(2:n))';

% =============================================================

end
