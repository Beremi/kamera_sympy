data = load("data/data.mat").input_data;
datasets = length(data);
bounds = [50 * 1e-3, 50 * 1e-3, 50 * 1e-3, 60 * pi / 180, 60 * pi / 180, 60 * pi / 180];
nvars = 6;

repeats = 100;
iters = zeros(datasets,repeats);
times = zeros(datasets,repeats);
true_errors = zeros(datasets,repeats);

% common parametes
max_iterations = 1000;
tolerance = 1e-4;

% CE parameters
max_iterations_CE = 100;
sample_size_CE = 4000;
elite_fraction = 0.05;
tolerance_CE = 1e-4;

%DE params
beta_min = 0.3; % Lower Bound of Scaling Factor
beta_max = 0.7; % Upper Bound of Scaling Factor
pCR = 0.5; % Crossover Probability
population_size = 4;
tolerance_DE = 1e+4;

tolerance_DEgrad = 1e-1;

% LH 
sample_size_LH = 400;
ratio_hess_step = 0.1;
n_newton_steps = 3;
max_iterations_newton = 100;

% pure Newton
maxit_newton = 100;

for i = 1:datasets
    fprintf("set %d:", i);
    F = @(x) valF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
    dF = @(x) gradF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
    ddF = @(x) hessF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
    for j = 1:repeats
        tic;
        %[x, it] = fmin_CE(F, zeros(1,nvars), diag(bounds.^2), sample_size_CE, tolerance_CE, elite_fraction, max_iterations_CE);
        %[x, it] = fmin_DEvec(F,  bounds, beta_min, beta_max, pCR, population_size, tolerance_DE, max_iterations);
        %[x, it] = fmin_DEvecNewton(F, dF, ddF, bounds, beta_min, beta_max, pCR, population_size, tolerance_DE, tolerance_DEgrad, max_iterations);
        %[x, it] = fmin_DE(F,  bounds, beta_min, beta_max, pCR, population_size, tolerance_DE, max_iterations);
        [x, it] = fmin_DENewton(F, dF, ddF, bounds, beta_min, beta_max, pCR, population_size, tolerance_DE, tolerance_DEgrad, max_iterations);
        %[x, it] = fmin_LHNewton(F, dF, ddF, bounds, sample_size_LH, ratio_hess_step, n_newton_steps, tolerance, max_iterations_newton);
        %[x, it] = fmin_Newton(F, dF, ddF, bounds, tolerance, maxit_newton);
        %[x, it] = fmin_Newton_rand(F, dF, ddF, bounds, tolerance, maxit_newton);

        time = toc;
        iters(i,j) = it;
        times(i, j) = time;
        fval_cur = F(x);
        true_errors(i, j) = fval_cur - data{i}.F_val_res;
        fprintf(".");
    end
    fprintf("\n");
end
mask = true_errors(:) > 1e-3;
[mean(mask)*100, 1000*mean(times(~mask)),1000*quantile(times(~mask), 0.99)] 
