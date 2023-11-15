function [x , it] = fmin_beres(S,f_presc,v,n,L,nn,epsilon)
%FMIN_BERES Summary of this function goes here
%   Detailed explanation goes here
F = @(x) valueF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);
dF = @(x) gradF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);
ddF = @(x) hessF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);


bounds = [50*1e-3,50*1e-3,50*1e-3,20*pi/180,20*pi/180,20*pi/180]';

points = bounds.*LatinHypercube(nn, 6);%

F_vals = zeros(nn,1);

for i = 1:nn
    F_loc = F(points(:,i));
    if  F_loc < 1
        [x,~] = mini_newton(points(:,i),F,dF,ddF,3,1e-1);
        for j = 4:6
            x(j) = mod(x(j),2*pi);
            if x(j) < -0.5
                x(j) = x(j) + 2*pi;
            end
            if x(j) > 0.5
                x(j) = x(j) - 2*pi;
            end
        end
        points(:,i) = x;
        F_loc = F(x);
    end
    F_vals(i) = F_loc;
end

% Sort the array in ascending order and get the sorting indices
[F_vals, sortIdx] = sort(F_vals, 'ascend');

% Use the sortIdx to sort the 'points' matrix column-wise
points = points(:, sortIdx);
for i = 1:ceil(nn/100)
    [x,it] = mini_newton(points(:,i),F,dF,ddF,100,epsilon);
    points(:,i) = x;
    F_vals(i) = F(x);
end
[F_vals, sortIdx] = sort(F_vals, 'ascend');

% Use the sortIdx to sort the 'points' matrix column-wise
points = points(:, sortIdx);
x = points(:,1);

for i = 4:6
    x(i) = mod(x(i),2*pi);
    if x(i) < -0.5
        x(i) = x(i) + 2*pi;
    end
    if x(i) > 0.5
        x(i) = x(i) - 2*pi;
    end
end
end


function [x,it] = mini_newton(x0,F,dF,ddF,maxit,eps)
x = x0;
for it = 1:maxit
    g = dF(x);
    H = ddF(x);
    g_norm = norm(g);
    g = H\g;
    d= -g;
    alpha = goldenSectionLineSearch(x, d, F, 0, 1, 1e-3);
    x = x + alpha*d;
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
