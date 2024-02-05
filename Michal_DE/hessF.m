function [dF] = hessF(x,S,f_presc,v,n,L)
% calculation of the gradient of the cost function F (sensitivity analysis)
%
%input:     X,Y,Z,alpha,beta,gamma - skalary 
%               alpha,beta,gamma - inff radians
%           S - vekt. 15x1 - souradnice bodu kalibracniho terce
%           f_presc - vekt. 10x1 - predepsany obraz
%           v - vekt. 3x1
%           n - vekt. 3x1
%           L - skalar - vzdalenost mezi objektivem a TCP bodem
%
%output:    dF - vekt. 6x1
%pomocne:   f - vekt. 10x1 - vypocteny obraz pro danou polohu kamery X,Y,Z a jeji orientaci alpha,beta,gamma  
%
%vse ve formatu double 

ff = 0.008; %focal length
X=x(:,1);Y=x(:,2);Z=x(:,3);alpha=x(:,4);beta=x(:,5);gamma=x(:,6);
dF=hessF_python2(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L,ff);
% n = length(X);
% for i=1:6
%     for j=1:6
%         dF{i,j} = reshape(dF{i,j},1,1,n);
%     end
% end
% dF=cell2mat(dF);