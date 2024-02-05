function [optimal_point, iterations] = fmin_CE(objective_func, initial_mean, initial_covariance,...
    sample_size, convergence_threshold, elite_fraction, max_iterations)
% FMIN_CE Performs minimization using the Cross Entropy method
%
% This function minimizes an objective function using the Cross Entropy (CE) method,
% which iteratively updates a distribution of points towards regions of minimum value.
%
% Parameters:
% - objective_func: Handle to the objective function to minimize.
% - initial_mean: Initial mean vector of the search distribution.
% - initial_covariance: Initial covariance matrix of the search distribution.
% - sample_size: Number of samples to draw in each iteration.
% - convergence_threshold: Threshold for the convergence criterion based on the covariance matrix.
% - elite_fraction: (Optional) Fraction of samples to use for updating the distribution. Default is 0.05.
% - max_iterations: (Optional) Maximum number of iterations to perform. Default is 100.
%
% Returns:
% - optimal_point: The point with the lowest objective function value found.
% - iterations: The number of iterations performed.

% Set default values for optional parameters
if nargin < 6
    elite_fraction = 0.05; % Default elite fraction
end
if nargin < 7
    max_iterations = 100; % Default maximum number of iterations
end

% Number of variables to optimize
num_vars = length(initial_mean);

% Calculate the number of elite points based on the elite fraction
num_elite_points = ceil(sample_size * elite_fraction);

% Decompose initial covariance matrix for sampling
L = chol(initial_covariance);

% Generate initial set of points based on the initial mean and covariance
sample_points = randn(sample_size, num_vars) * L + initial_mean;

% Evaluate the objective function at all points
objective_values = objective_func(sample_points);

% Sort points by their objective function values in ascending order
[~, sorted_indices] = sort(objective_values, 'ascend');
sample_points = sample_points(sorted_indices, :);

% Select elite points (points with lowest objective values)
elite_points = sample_points(1:num_elite_points, :);

% Update mean and covariance based on elite points
updated_mean = mean(elite_points);
updated_covariance = cov(elite_points);

% Decompose updated covariance for sampling
L = chol(updated_covariance);

% Iteration loop
for iterations = 1:max_iterations
    % Sample new points based on updated mean and covariance
    sample_points = randn(sample_size, num_vars) * L + updated_mean;
    objective_values = objective_func(sample_points);
    
    % Sort new points by objective function values
    [~, sorted_indices] = sort(objective_values, 'ascend');
    sample_points = sample_points(sorted_indices, :);
    elite_points = sample_points(1:num_elite_points, :);
    
    % Update mean and covariance with new elite points
    updated_mean = mean(elite_points);
    updated_covariance = cov(elite_points);
    L = chol(updated_covariance);
    
    % Check for convergence based on the sum of diagonal elements of the covariance matrix
    if sqrt(sum(diag(updated_covariance))) < convergence_threshold
        break;
    end
end

% Output the best point found and the number of iterations
optimal_point = sample_points(1, :);

end
