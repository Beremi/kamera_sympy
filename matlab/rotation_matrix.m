function [R] = rotation_matrix(alpha, beta, gamma)
%input: alpha,beta,gamma - v radianech
%
%output:    R - mat. 3x3
%
%vse ve formatu double
%
% matice rotace 

R_alpha=[1 0 0; 0 cos(alpha) -sin(alpha); 0 sin(alpha) cos(alpha)];
R_beta=[cos(beta) 0 sin(beta); 0 1 0; -sin(beta) 0 cos(beta)];
R_gamma=[cos(gamma) -sin(gamma) 0; sin(gamma) cos(gamma) 0; 0 0 1];
R=R_alpha*R_beta*R_gamma;


end