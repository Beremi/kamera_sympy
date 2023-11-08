function [x,it]=fmin_sd(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L,epsilon) 
% nD minimization by steepest descent (gradient) method
%
%input:     X,Y,Z,alpha,beta,gamma - skalary 
%               alpha,beta,gamma - in radians
%           S - vekt. 15x1 - souradnice bodu kalibracniho terce
%           f_presc - vekt. 10x1 - predepsany obraz
%           v - vekt. 3x1
%           n - vekt. 3x1
%           L - skalar - vzdalenost mezi objektivem a TCP bodem
%           epsilon - skalar
%
%output:    x - vekt. 6x1
%           it - skalar
%pomocne:   f - vekt. 10x1 - vypocteny obraz pro danou polohu kamery X,Y,Z a jeji orientaci alpha,beta,gamma  
%           x,xn - vekt. 6x1
%           g - vekt. 6x1
%           gnorm - skalar
%           d, t - skalar
%           fxn,fx - skalar
%vse ve formatu double 

epsil=1e-12;

koef=0.9;
alph=0.1;

x=[X,Y,Z,alpha,beta,gamma]';
it=1;

n0=n;

% [f,df] = value_f_df(x(1),x(2),x(3),x(4),x(5),x(6),S,v,n,L);  

g=gradF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);  
gnorm=norm(g);
if norm(g)<=epsil
    gnorm=epsil;
end

d=-g/gnorm;

% R=rotation_matrix(x(4),x(5),x(6));
% n=R*n0;

fx=valueF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);
fx0=fx;

% d(4:6)=d(4:6)*1e-2;%*1e-3;
%line search
t=golden_section(S,f_presc,v,n,L,0,1,d,1e-3,x(1),x(2),x(3),x(4),x(5),x(6));
% d(4:6)=d(4:6)*1e-1;%*1e-3;

%Length step modification
h=t*d;
lambda=1;
xn=x+lambda*h;

fxn=valueF(xn(1),xn(2),xn(3),xn(4),xn(5),xn(6),S,f_presc,v,n,L);

while fxn > fx+alph*lambda*g'*h && lambda>1e-4
    lambda=lambda*koef;
    xn=x+lambda*h;
    
%     R=rotation_matrix(xn(4),xn(5),xn(6));
%     n=R*n0;

    fxn=valueF(xn(1),xn(2),xn(3),xn(4),xn(5),xn(6),S,f_presc,v,n,L);
end

while abs(fxn-fx)/fx0 >= epsilon && it<100 %fxn/fx0 >= epsilon && it<1000 %norm(xn-x) >= epsilon && it<2000
    
    it = it + 1;
    x = xn; 
    fx = fxn;
    
    g=gradF(x(1),x(2),x(3),x(4),x(5),x(6),S,f_presc,v,n,L);
    gnorm=norm(g);
    if norm(g)<=epsil
        gnorm=epsil;
    end
        
    d=-g/gnorm;
    
    %   d(4:6)=d(4:6)*1e-2;%*1e-3;
    %line search
    t=golden_section(S,f_presc,v,n,L,0,0.3,d,1e-3,x(1),x(2),x(3),x(4),x(5),x(6));
    %   d(4:6)=d(4:6)*1e-1;%*1e-3;
    
    %Length step modification
    h=t*d;
    lambda=1;
    xn=x+lambda*h;
    
    fxn=valueF(xn(1),xn(2),xn(3),xn(4),xn(5),xn(6),S,f_presc,v,n,L);
    
    while fxn > fx+alph*lambda*g'*h && lambda>1e-4
        lambda=lambda*koef;
        xn=x+lambda*h;
        
%         R=rotation_matrix(xn(4),xn(5),xn(6));
%         n=R*n0;
        
        fxn=valueF(xn(1),xn(2),xn(3),xn(4),xn(5),xn(6),S,f_presc,v,n,L);
    end
    
end

x = xn;
it
abs(fxn-fx)/fx0

end