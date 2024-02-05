function [optimal_point, iterations] = fmin_DEvec(objective_func, bounds, beta_min, beta_max, pCR, population_size, convergence_threshold, max_iterations)
%% Differential Evolution (DE) Optimization
%
% This function minimizes an objective function using the Differential Evolution (DE) algorithm.
% DE is a stochastic population-based optimization algorithm.
%
% Parameters:
% - objective_func: Handle to the objective function to minimize.
% - bounds: Boundaries for the variables.
% - beta_min: Minimum scaling factor for mutation.
% - beta_max: Maximum scaling factor for mutation.
% - pCR: Crossover probability.
% - population_size: Number of individuals in the population.
% - convergence_threshold: Threshold for convergence based on the change in cost.
% - max_iterations: Maximum number of iterations.
%
% Returns:
% - optimal_point: The best point found.
% - iterations: The number of iterations performed.

%% Initialization
num_vars = size(bounds, 2); % Assuming bounds is a 1xN
position_current = 2 * (rand(population_size, num_vars) - 0.5) .* bounds;
cost_current = objective_func(position_current);

[~, best_idx] = min(cost_current);
best_solution = position_current(best_idx, :);

best_cost_trace = zeros(max_iterations, 1); % Track best cost at each iteration
best_cost_trace(1) = inf; % Initial high value to ensure first cost is recorded
iterations = 0;

test_cost_current = geomean(cost_current);
test_cost_previous = inf;

%% DE Main Loop
while ((abs(test_cost_current - test_cost_previous) > convergence_threshold) && (iterations < max_iterations))

    [idx_a, idx_b, idx_c] = rand_mutations(population_size);

    % Mutation
    beta = rand(population_size, num_vars) * (beta_max - beta_min) + beta_min;
    mutant_vectors = position_current(idx_a,:) + beta .* (position_current(idx_b,:) - position_current(idx_c,:));
    mutant_vectors = max(mutant_vectors, -bounds);
    mutant_vectors = min(mutant_vectors, bounds);

    % Crossover
    trial_vectors = position_current;
    crossover_mask = rand(population_size, num_vars) <= pCR;
    % Ensure at least one variable is changed
    crossover_mask(randi([1, num_vars], population_size, 1) + (0:population_size-1)'*num_vars) = true;
    trial_vectors(crossover_mask) = mutant_vectors(crossover_mask);

    % Selection
    new_costs = objective_func(trial_vectors);
    improvement_mask = new_costs < cost_current;
    cost_current(improvement_mask) = new_costs(improvement_mask);
    position_current(improvement_mask, :) = trial_vectors(improvement_mask, :);

    [current_best_cost, best_idx] = min(cost_current);
    best_solution = position_current(best_idx, :);

    iterations = iterations + 1;
    best_cost_trace(iterations) = current_best_cost;

    test_cost_previous = test_cost_current;
    test_cost_current = geomean(cost_current);
end

optimal_point = best_solution;

end

function [a, b, c] = rand_mutations(n)
% generating random indexes such in each row its uniquee three values
% (a,b,c) and also different from idex of row i
% its done using generating numbers always from range [1 n-k], where k
% is number of indexes which we are avoiding (a -> k=1, b -> k=2, ...)
% then we need go through avoiding indexes in ascending order and
% add +1 to thhe index if its same or greater

idx = (1:n)';

% a
a = randi([1 n-1], n, 1);
a(a>=idx) = a(a>=idx)+1;

% b
b = randi([1 n-2], n, 1);
b(b>=min(idx,a)) = b(b>=min(idx,a))+1;
b(b>=max(idx,a)) = b(b>=max(idx,a))+1;

%c
c = randi([1 n-3], n, 1);
% here the middle index is bit trickier
d1 = min(idx,a);
d2 = min(a,b);
d3 = min(idx,b);
dd1 = max(d1,d2);
middle_idx = max(dd1, d3);
min_idx = min(d1,d2);
max_idx = max(max(idx,a),b);
c(c>=min_idx) = c(c>=min_idx)+1;
c(c>=middle_idx) = c(c>=middle_idx)+1;
c(c>=max_idx) = c(c>=max_idx)+1;
end
