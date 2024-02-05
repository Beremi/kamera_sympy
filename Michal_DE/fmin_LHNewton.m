function [x , it] = fmin_LHNewton(F, dF, ddF, bounds, n_samples, ratio_hess_step, n_newton_steps, epsilon, maxit_newton)
%FMIN_BERES Summary of this function goes here
%   Detailed explanation goes here
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');

points = bounds.*LatinHypercube(n_samples, 6)';
F_vals = F(points);
[~, sortIdx] = sort(F_vals, 'ascend');
points = points(sortIdx,:);

nn_hess = ceil(n_samples*ratio_hess_step);
z = points(1:nn_hess,:);
for j=1:n_newton_steps
    grads = dF(z);
    hesses = ddF(z);
    
    for i=1:nn_hess
        z(i,:) = z(i,:) - (hesses(:,:,i)\grads(:,i))';
    end
end
points = z;

F_vals = F(points);

% Sort the array in ascending order and get the sorting indices
[F_vals, sortIdx] = sort(F_vals, 'ascend');

% Use the sortIdx to sort the 'points' matrix column-wise
points = points(sortIdx,:);
for i = 1:1
    [x,it] = mini_newton(points(i,:),F,dF,ddF,maxit_newton,epsilon);
    points(i,:) = x;
    F_vals(i) = F(x);
end
[~, sortIdx] = sort(F_vals, 'ascend');

% Use the sortIdx to sort the 'points' matrix column-wise
points = points(sortIdx,:);
x = points(1,:);

warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:SingularMatrix');

end


function [x,it] = mini_newton(x0,F,dF,ddF,maxit,eps)
x = x0;
for it = 1:maxit
    g = dF(x);
    H = ddF(x);
    g_norm = norm(g);
    g = H\g;
    d= -g;
    alpha = goldenSectionLineSearch(x, d', F, -1, 1, 1e-1);
    x = x + alpha*d';
    if g_norm < eps
        break
    end
end

end

function alpha = goldenSectionLineSearch(x, d, F, a, b, eps)
% Define the golden ratio
rho = (sqrt(5) - 1) / 2;

% Define the end points of the initial interval
a0 = a;
b0 = b;

% Calculate the two interior points
a1 = b0 - rho * (b0 - a0);
b1 = a0 + rho * (b0 - a0);

% Evaluate the function at the interior points
F_a1 = F(x + a1 * d);
F_b1 = F(x + b1 * d);

% Iterate until the interval is smaller than the tolerance
while (abs(b0 - a0) > eps)
    if (F_a1 < F_b1)
        % If the function value at a1 is less, we choose the interval [a0, b1]
        b0 = b1;
        b1 = a1;
        F_b1 = F_a1;
        a1 = b0 - rho * (b0 - a0);
        F_a1 = F(x + a1 * d);
    else
        % If the function value at b1 is less, we choose the interval [a1, b0]
        a0 = a1;
        a1 = b1;
        F_a1 = F_b1;
        b1 = a0 + rho * (b0 - a0);
        F_b1 = F(x + b1 * d);
    end
end

% Choose the best estimate for the minimum
if F_a1 < F_b1
    alpha = (a0 + b1) / 2;
else
    alpha = (a1 + b0) / 2;
end
end

function samples = LatinHypercube(num_samples, num_dimensions)
    % Preallocate the samples array
    samples = zeros(num_samples, num_dimensions);
    
    % Generate the intervals
    intervals = linspace(-1, 1, num_samples+1);
    
    % Create a Latin Hypercube Sample
    for i = 1:num_dimensions
        % Permute the intervals for the current dimension
        permuted_intervals = intervals(randperm(num_samples));
%         [~,idx] = sort(rand(num_samples,1));
%         permuted_intervals = intervals(idx);
        % Take the midpoints of these intervals to ensure one sample per interval
        samples(:, i) = permuted_intervals(1:num_samples)' + diff(intervals(1:2))/2;
    end
    
    samples = samples' + (rand(num_dimensions,num_samples)-0.5)*(2/num_samples);
end
