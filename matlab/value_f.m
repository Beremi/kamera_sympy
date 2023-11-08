function [f] = value_f(X,Y,Z,alpha,beta,gamma,S,v,n,L)
% calculation of the value of the vector function f:R^6 -> R^10
%
%input:     X,Y,Z,alpha,beta,gamma - skalary
%               alpha,beta,gamma - in radians
%           S - vekt. 15x1 - souradnice bodu kalibracniho terce
%           v - vekt. 3x1
%           n - vekt. 3x1
%           L - skalar - vzdalenost mezi objektivem a TCP bodem
%
%output:    f - vekt. 10x1 - vypocteny obraz pro danou polohu kamery X,Y,Z a jeji orientaci alpha,beta,gamma
%
%vse ve formatu double

ff = 0.008; %focal length

f = zeros(10,1);

R=rotation_matrix(alpha,beta,gamma);
n=R*n;



f = zeros(10,1);

for i=1:5 
    [res1, res2] = f_value_single_python(alpha, beta, gamma, X, Y, Z, L, v(1), v(2), v(3), n(1), n(2), n(3), ff, S(3*i-2), S(3*i-1), S(3*i));
    f(2*i-1) = res1;
    
    f(2*i) = res2;
end

f = f.*1e3;

% f-f_orig
end
