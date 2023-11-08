function [dF] = gradF(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L)
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

dF = zeros(6,1);

% df_X = zeros(10,1);
% df_Y = zeros(10,1);
% df_Z = zeros(10,1);
% df_alpha = zeros(10,1);
% df_beta = zeros(10,1);
% df_gamma = zeros(10,1);



%numericka aproximace
f = value_f(X,Y,Z,alpha,beta,gamma,S,v,n,L);
num_grad_f=zeros(6*10,8);
x=[X,Y,Z,alpha,beta,gamma];
st=1e-1;

step=st.^9;

for i=1:6
    
    x_step=x;
    x_step(i)=x_step(i)+step;
    
    f_step=value_f(x_step(1),x_step(2),x_step(3),x_step(4),x_step(5),x_step(6),S,v,n,L);
    
    num_grad_f(6*(i-1)+1:6*(i-1)+10,1)=(f_step-f)/step;
end

df=num_grad_f;

df_X = df(1:10);
df_Y = df(11:20);
df_Z = df(21:30);
df_alpha = df(31:40);
df_beta = df(41:50);
df_gamma = df(51:60);


f = value_f(X,Y,Z,alpha,beta,gamma,S,v,n,L);

for i=1:5
    dF(1) = dF(1) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_X(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_X(2*i).*1e3; 
    dF(2) = dF(2) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_Y(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_Y(2*i).*1e3; 
    dF(3) = dF(3) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_Z(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_Z(2*i).*1e3; 
    dF(4) = dF(4) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_alpha(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_alpha(2*i).*1e3; 
    dF(5) = dF(5) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_beta(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_beta(2*i).*1e3; 
    dF(6) = dF(6) + 2.*(f(2*i-1)-f_presc(2*i-1)).*df_gamma(2*i-1).*1e3 + 2.*(f(2*i)-f_presc(2*i)).*df_gamma(2*i).*1e3; 
end


% dF(4:6) = 1e1.*dF(4:6);


% %numericka aproximace
% f = valueF(X,Y,Z,alpha,beta,gamma,S,f_presc,v,n,L);
% num_grad_F=zeros(6,10);
% x=[X,Y,Z,alpha,beta,gamma];
% st=1e-1;
% 
% for k=1:10
%     step=st.^k;
%     
%     for i=1:6
%         
%         x_step=x;
%         x_step(i)=x_step(i)+step;
%         
%         f_step=valueF(x_step(1),x_step(2),x_step(3),x_step(4),x_step(5),x_step(6),S,f_presc,v,n,L);
%         
%         num_grad_F(i,k)=(f_step-f)/step;
%     end
%     
% end
%    
% [dF num_grad_F]
% [dF num_grad_F(:,10)]