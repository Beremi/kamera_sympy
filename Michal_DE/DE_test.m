%verze 3. 8. 2022
%zakladni jednotky v milimetrech
images = table2array(readtable('body.xlsx', 'Range', 'A1:J102', 'ReadVariableNames', 0));
camera_coordinates = table2array(readtable('souradnice.xlsx', 'Range', 'A1:F102', 'ReadVariableNames', 0));

beta_min = 0.3; % Lower Bound of Scaling Factor
beta_max = 0.7; % Upper Bound of Scaling Factor
pCR = 0.5; % Crossover Probability

%Prevod mezi globalnim souradnym systemem a souradnym systemem ELVACu
X = camera_coordinates;
X(:, 4:6) = X(:, 4:6) .* pi / 180;

aa = images(:, 1:2);
bb = images(:, 3:4);
cc = images(:, 5:6);
dd = images(:, 7:8);
ee = images(:, 9:10);

%transformace Bail-Beremlijski system -> ELVAC system
Trasnform_matrix = [0 -1 0; 1 0 0; 0 0 -1];
Trasnform_matrix_rotation = [0 1 0; -1 0 0; 0 0 1];

%kalibracni terc
S = [0; 0; 0; 0.05; 0.05; 0; 0.05; -0.05; 0; -0.05; -0.05; 0; -0.05; 0.05; 0] .* 1e3;

v = [0; 0; -1];
n = [0; 1; 0];
L = 9.5 * 1e-3;

F_val = [];

input_data = {};
for j = 1:102
    if j==92
        continue
    end
    instance_data = struct();
    x = X(j, :)';
    x = [Trasnform_matrix * x(1:3); Trasnform_matrix_rotation * x(4:6)];
    L = min([309.5 * 1e-3 309.5 * 1e-3 + x(3) .* 1e-3]);
    x(3) = max([x(3), 0]);
    SS = [2592/2; 1944/2];
    
    A = aa(j, :)' - SS;
    B = bb(j, :)' - SS;
    C = cc(j, :)' - SS;
    D = dd(j, :)' - SS;
    E = ee(j, :)' - SS;
    
    f_presc = [A .* 2.2/1e6; B .* 2.2/1e6; C .* 2.2/1e6; D .* 2.2/1e6; E .* 2.2/1e6] .* 1e3;
    f_presc([3:4; 7:8]) = f_presc([7:8; 3:4]); %prohozene osy => B<-->D
    instance_data.S = S .* 1e-3;
    instance_data.f_presc = f_presc;
    instance_data.v = v;
    instance_data.n = n;
    instance_data.L = L;
    instance_data.idx = j;
    

    [x, fout, it] = DEvec(S .* 1e-3, f_presc, v, n, L, beta_min, beta_max, pCR, 1e-5);
    disp(it)
    F_val1(j) = valueF_DE(x, S .* 1e-3, f_presc, v, n, L);
    CostFunction = @(x) valF(x, S .* 1e-3, f_presc, v, n, L);
    bounds = [50 * 1e-3, 50 * 1e-3, 50 * 1e-3, 20 * pi / 180, 20 * pi / 180, 60 * pi / 180];
    [x, it2] = fmin_CE(CostFunction,zeros(1,6),diag(bounds.^2), 2000, 1e-10, 0.05,1000);
    disp(it2)
    %fmin_CE(objective_func, initial_mean, initial_covariance, sample_size, convergence_threshold, elite_fraction, max_iterations)
    F_val2(j) = valueF_DE(x, S .* 1e-3, f_presc, v, n, L);
    instance_data.F_val_res = F_val2(j);
    input_data= [input_data {instance_data}];
    %disp([[out(1:3) * 1e3, out(4:6)]; x']);
end
% figure
plot(F_val1)
hold on
plot(F_val2)
set(gca,'yscale','log')
