function [out, fout, it] = DE(S, f_presc, v, n, L, beta_min, beta_max, pCR, eps)
%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA107
% Project Title: Implementation of Differential Evolution (DE) in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
%
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
%
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
% Suppress the specific warning
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
% Your code here...


%% Problem Definition

CostFunction = @(x) valueF_DE(x, S, f_presc, v, n, L);
nVar = 6; % Number of Decision Variables
VarSize = [1 nVar]; % Decision Variables Matrix Size

dF = @(x) gradF(x(1), x(2), x(3), x(4), x(5), x(6), S, f_presc, v, n, L);
ddF = @(x) hessF(x(1), x(2), x(3), x(4), x(5), x(6), S, f_presc, v, n, L);

%% DE Parameters

MaxIt = 100; % Maximum Number of Iterations
nPop = nVar; % Population Size

%% Initialization
bounds = [50 * 1e-3, 50 * 1e-3, 50 * 1e-3, 20 * pi / 180, 20 * pi / 180, 20 * pi / 180];
empty_individual.Position = [];
empty_individual.Cost = [];

BestSol.Cost = inf;

pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop
    
    pop(i).Position = [unifrnd(-50 * 1e-3, 50 * 1e-3, [1 3]) unifrnd(-20 * pi / 180, 20 * pi / 180, [1 3])];
    pop(i).Cost = CostFunction([pop(i).Position]);
    
    if pop(i).Cost < BestSol.Cost
        BestSol = pop(i);
    end
    
end

BestCost = zeros(MaxIt, 1);
GradSize = zeros(MaxIt, 1);
BestCost(1) = 1000;
%% DE Main Loop
it = 0;
BC = 1000;
BC_old = 1e16;
g_best = inf * ones(MaxIt, 1);
g = g_best;
normg = 1;

while (((norm(BC-BC_old) > eps) || (normg > 1e-3)) && (it < MaxIt))
    
    for i = 1:nPop
        
        x = pop(i).Position;
        
        A = randperm(nPop);
        
        A(A == i) = [];
        
        a = A(1);
        b = A(2);
        c = A(3);
        
        % Mutation
        beta = unifrnd(beta_min, beta_max, VarSize);
        y = pop(a).Position + beta .* (pop(b).Position - pop(c).Position);
        y = max(y, -bounds);
        y = min(y, bounds);
        
        % Crossover
        z = zeros(size(x));
        j0 = randi([1 numel(x)]);
        
        for j = 1:numel(x)
            
            if j == j0 || rand <= pCR
                z(j) = y(j);
            else
                z(j) = x(j);
            end
            
        end
        
        [z, g] = newton_step(z, dF, ddF, 0.95);
        
        NewSol.Position = z;
        NewSol.Cost = CostFunction(NewSol.Position);
        
        if NewSol.Cost < pop(i).Cost
            pop(i) = NewSol;
            
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
                g_best = g;
            end
            
        end
    end
    
    it = it + 1;
    
%     for newton_i = 1:10
%         [BestSol.Position, g_best] = newton_step(BestSol.Position, dF, ddF, 0.95);
%     end
%     BestSol.Cost = CostFunction(BestSol.Position);

    
    normg = norm(g_best);
    % Update Best Cost
    BestCost(it) = BestSol.Cost;
    GradSize(it) = normg;
    BC_old = BC;
    BC = BestCost(it);
    
end

out = BestSol.Position; %init'+BestSol.Position;
fout = BestSol.Cost;
%% Show Results

% figure;
% %plot(BestCost);
% semilogy(BestCost, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;
% Optionally, re-enable the warning after your code
warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:SingularMatrix');
end

function [x_new, g] = newton_step(x, dF, ddF,alpha)
g = dF(x');
H = ddF(x');
x_new = x - alpha*(H\g)';

end
