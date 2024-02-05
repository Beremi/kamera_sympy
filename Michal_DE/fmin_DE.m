function [out, it] = fmin_DE(F, bounds, beta_min, beta_max, pCR, nPop, tolerance, MaxIt)
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
nVar = length(bounds); % Number of Decision Variables
VarSize = [1 nVar]; % Decision Variables Matrix Size


%% DE Parameters

%% Initialization
empty_individual.Position = [];
empty_individual.Cost = [];

BestSol.Cost = inf;

pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop
    
    pop(i).Position = 2*(rand(1, nVar) - 0.5).*bounds;
    pop(i).Cost = F(pop(i).Position);
    
    if pop(i).Cost < BestSol.Cost
        BestSol = pop(i);
    end
    
end

BestCost = zeros(MaxIt, 1);
BestCost(1) = 1000;
%% DE Main Loop
it = 0;
cur_tol_crit = geomean([pop.Cost]);
old_tol_crit = inf;

while ((abs(cur_tol_crit-old_tol_crit) > tolerance) && (it < MaxIt)) 
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

        NewSol.Position = z;
        NewSol.Cost = F(NewSol.Position);
        
        if NewSol.Cost < pop(i).Cost
            pop(i) = NewSol;
            
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
            
        end
    end
    
    it = it + 1;
    
    % Update Best Cost
    BestCost(it) = BestSol.Cost;
    old_tol_crit = cur_tol_crit;
    cur_tol_crit = geomean([pop.Cost]);
    
end

out = BestSol.Position; %init'+BestSol.Position;

warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:SingularMatrix');
end
