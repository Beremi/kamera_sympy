function [optimal_solution, iterations] = fmin_DEvecNewton(objective_function, gradient_objf, hessian_objf, ...
    bounds, beta_min, beta_max, pCR, population_size, tolerance_cost, tolerance_gradient, max_iterations)
%% Differential Evolution with Newton's Method Optimization Algorithm
%
% Parameters:
% - objective_function: Function to be minimized.
% - gradient_function: Function that computes the gradient of the objective function.
% - hessian_function: Function that computes the Hessian of the objective function.
% - beta_min, beta_max: Parameters controlling the mutation scale.
% - pCR: Crossover probability.
% - population_size: Number of individuals in the population.
% - tolerance_cost: Convergence tolerance based on the cost improvement.
% - tolerance_gradient: Convergence tolerance based on the gradient norm.
% - max_iterations: Maximum number of iterations.
%
% Returns:
% - optimal_solution: The best solution found.
% - iterations: The number of iterations performed.

% Suppress the specific warning
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');

%% Initialization
nVar = numel(bounds); % Assuming 'bounds' is globally defined or passed in some way
position_current = 2 * (rand(population_size, nVar) - 0.5) .* bounds;
cost_current = objective_function(position_current);
grad_current = gradient_objf(position_current); % Initial gradient computation

[~, best_idx] = min(cost_current);
BestSol = position_current(best_idx,:);
best_grad = grad_current(:,best_idx);

%% DE Main Loop
iterations = 0;
test_cost_curr = geomean(cost_current);
test_cost_old = inf;

while (((abs(test_cost_curr - test_cost_old) > tolerance_cost) || (norm(best_grad) > tolerance_gradient)) && ...
        (iterations < max_iterations))

    % indexes for mutation to comply with i!=a!=b!=c
    [a, b, c] = rand_mutations(population_size);

    % Mutation
    beta = rand(population_size, nVar) * (beta_max - beta_min) + beta_min;
    mutated_vectors = position_current(a,:) + beta .* (position_current(b,:) - position_current(c,:));
    mutated_vectors = min(max(mutated_vectors, -bounds), bounds);

    % Crossover
    trial_vectors = position_current;
    j_rand = rand(population_size, nVar) <= pCR;
    j_rand(randi([1, nVar], population_size, 1) + (0:population_size-1)'*nVar) = true;
    trial_vectors(j_rand) = mutated_vectors(j_rand);

    % Selection based on Objective Function Improvement
    new_costs = objective_function(trial_vectors);
    improvement_flags = new_costs < cost_current;
    position_current(improvement_flags, :) = trial_vectors(improvement_flags, :);
    cost_current(improvement_flags) = new_costs(improvement_flags);

    % Newton's Method Update
    trial_vectors = position_current;
    grads = gradient_objf(trial_vectors);
    hessians = hessian_objf(trial_vectors);

    for i = 1:population_size
        dir_newton = - (hessians(:,:,i) \ grads(:,i))';
        all_point_search_newton = trial_vectors(i,:)+dir_newton .* linspace(-1,1,8)';
        [~, idx_newton_line] = min(objective_function(all_point_search_newton));
        trial_vectors(i,:) = all_point_search_newton(idx_newton_line,:);
    end

    new_costs = objective_function(trial_vectors);
    improvement_flags = new_costs < cost_current;
    position_current(improvement_flags, :) = trial_vectors(improvement_flags, :);
    cost_current(improvement_flags) = new_costs(improvement_flags);
    grad_current(:,improvement_flags) = grads(:,improvement_flags);

    [current_best_cost, best_idx] = min(cost_current);
    BestSol = position_current(best_idx,:);
    best_grad = grad_current(:,best_idx);

    iterations = iterations + 1;
    test_cost_old = test_cost_curr;
    test_cost_curr = geomean(cost_current);
end

optimal_solution = BestSol;

% Re-enable warnings
warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:SingularMatrix');
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
