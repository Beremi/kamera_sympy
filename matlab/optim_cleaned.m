images=table2array(readtable('body.xlsx','Range','A1:J102','ReadVariableNames',0));
camera_coordinates=table2array(readtable('souradnice.xlsx','Range','A1:F102','ReadVariableNames',0));
X=camera_coordinates;

X(:,4:6)=X(:,4:6).*pi/180;

XX = X;

aa=images(:,1:2);
bb=images(:,3:4);
cc=images(:,5:6);
dd=images(:,7:8);
ee=images(:,9:10);

Trasnform_matrix = [0 -1 0; 1 0 0; 0 0 -1];
Trasnform_matrix_rotation = [0 1 0; -1 0 0; 0 0 1];

S=[0;0;0;0.05;0.05;0;0.05;-0.05;0;-0.05;-0.05;0;-0.05;0.05;0].*1e3;

v=[0;0;-1];
n=[0;1;0];

L = 12.5.*1e-3;
L = 9.5*1e-3;

F_val=[];
for j=1:102 
    x=X(j,:)';
    x=[Trasnform_matrix*x(1:3);Trasnform_matrix_rotation*x(4:6)];
    
    L =min([309.5*1e-3 309.5*1e-3 + x(3).*1e-3]);
    x(3) = max([x(3),0]);
 
    SS=[2592/2;1944/2];

    A=aa(j,:)'-SS;
    B=bb(j,:)'-SS;
    C=cc(j,:)'-SS;
    D=dd(j,:)'-SS;
    E=ee(j,:)'-SS;

    f_presc=[A.*2.2/1e6;B.*2.2/1e6;C.*2.2/1e6;D.*2.2/1e6;E.*2.2/1e6].*1e3;
    f_presc([3:4;7:8]) = f_presc([7:8;3:4]); 

    x01= x+[1,1,1,0.00001,0.00001,0.00001]';
    
    [x , it] = fmin_beres(S.*1e-3,f_presc,v,n,L,1000,1e-6);
    x(1:3)=x(1:3)*1e3;
    
    [f] = value_f(x(1).*1e-3,x(2).*1e-3,x(3).*1e-3,x(4),x(5),x(6),S.*1e-3,v,n,L);
    F_val = [F_val valueF(x(1).*1e-3,x(2).*1e-3,x(3).*1e-3,x(4),x(5),x(6),S.*1e-3,f_presc,v,n,L)];    
end
