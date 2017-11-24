function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

regularization_loss = (lambda / 2) * sum(sum(Theta.^2, 2)) + (lambda / 2) * sum(sum(X.^2, 2));

J = (1/2) * sum(sum((X * Theta' - Y).^2 .* R)) + regularization_loss;
% for each movie, calculate the error of the model rating prediction for each users that rated that movie, and
% weight each x vector value by errors across all users by multiplying the movie x vector by those errors
for i = 1:num_movies,
  users_who_rated_movie = find(R(i, :) == 1);
  Theta_temp = Theta(users_who_rated_movie, :);
  Y_temp = Y(i, users_who_rated_movie);
  pred = Theta_temp * X(i, :)';
  error = pred - Y_temp';
  % error - 2x1, Theta_temp - 2x3
  X_grad(i, :) = (Theta_temp' * error)' + X(i, :) .* lambda;
endfor

% for each user, calculate the error of the model rating prediction for each movie that user rated, and multiply the
% user thetas (weights of x vector) by those errors
for j = 1:num_users,
  movies_rated_by_user = find(R(:, j) == 1);
  X_temp = X(movies_rated_by_user, :);
  Y_temp = Y(movies_rated_by_user, j);
  pred = X_temp * Theta(j, :)';
  error = pred - Y_temp;
  % error - 5x1, Theta_temp - 5x3
  Theta_grad(j, :) = error' * X_temp + Theta(j, :) .* lambda;
endfor

grad = [X_grad(:); Theta_grad(:)];

% for j = 1:num_users,
% endfor

end

