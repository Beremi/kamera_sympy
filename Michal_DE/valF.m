function [F] = valF(x,S,f_presc,v,n,L)
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
X=x(:,1);Y=x(:,2);Z=x(:,3);alpha=x(:,4);beta=x(:,5);gamma=x(:,6);
ff = 0.008; %focal length

F = valueF_python2(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L,ff);

end
