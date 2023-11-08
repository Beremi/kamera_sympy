function [F] = valueF(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L)
% calculation of the value of the cost function F
%
%input:     X,Y,Z,alpha,beta,gamma - skalary 
%               alpha,beta,gamma - in radians
%           S - vekt. 15x1 - souradnice bodu kalibracniho terce
%           f_presc - vekt. 10x1 - predepsany obraz
%           v - vekt. 3x1
%           n - vekt. 3x1
%           L - skalar - vzdalenost mezi objektivem a TCP bodem
%
%output:    F - skalar
%pomocne:   f - vekt. 10x1 - vypocteny obraz pro danou polohu kamery X,Y,Z a jeji orientaci alpha,beta,gamma  
%
%vse ve formatu double 

f = value_f(X,Y,Z,alpha,beta,gamma,S,v,n,L);

F = sum((f-f_presc).^2);

end
