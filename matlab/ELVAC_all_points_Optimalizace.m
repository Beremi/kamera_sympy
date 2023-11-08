%verze 3. 8. 2022

%zakladni jednotky v milimetrech

images=table2array(readtable('body.xlsx','Range','A1:J102','ReadVariableNames',0));

camera_coordinates=table2array(readtable('souradnice.xlsx','Range','A1:F102','ReadVariableNames',0));

%Prevod mezi globalnim souradnym systemem a souradnym systemem ELVACu, alpha=beta=0, gamma=-pi/2
%osa x je naopak oto�en�!  m�me prohozeny osy x a y a kolem y rotujeme opa�n�!
X=camera_coordinates;

X(:,4:6)=X(:,4:6).*pi/180;

XX = X;


aa=images(:,1:2);
bb=images(:,3:4);
cc=images(:,5:6);
dd=images(:,7:8);
ee=images(:,9:10);
    

%transformace Bail-Beremlijski syst�m -> ELVAC syst�m
Trasnform_matrix = [0 -1 0; 1 0 0; 0 0 -1];
Trasnform_matrix_rotation = [0 1 0; -1 0 0; 0 0 1];

%x=[0.0056;-0.0177;299.98;0.0005*pi/180;0.0034*pi/180;10.0005*pi/180];




%kalibracni terc
S=[0;0;0;0.05;0.05;0;0.05;-0.05;0;-0.05;-0.05;0;-0.05;0.05;0].*1e3;

% for k=1:5
%     S(k*3-2:k*3) = (Trasnform_matrix*S(k*3-2:k*3))';
% end

v=[0;0;-1];
n=[0;1;0];

L = 12.5.*1e-3;
L = 9.5*1e-3;


fileID = fopen('Table.txt','w');
fileID2 = fopen('Table_calib.txt','w');

for j=1:102
       
    x=X(j,:)';
    x=[Trasnform_matrix*x(1:3);Trasnform_matrix_rotation*x(4:6)];
        
    %L mus� b�t upraveno dle z
    L =min([309.5*1e-3 309.5*1e-3 + x(3).*1e-3]);
    x(3) = max([x(3),0]);
    
%     y=x; y(3)=L;
%     [x_temp] = transformace(zeros(1,6),[zeros(3,1);x(4:6)]',y');
%     x_temp(3)=x(3);
%     
%     x = x_temp;
     SS=[2592/2;1944/2];
%     
     A=aa(j,:)'-SS;
     B=bb(j,:)'-SS;
     C=cc(j,:)'-SS;
     D=dd(j,:)'-SS;
     E=ee(j,:)'-SS;
%     x01=[0.1,0.1,0.1,0.1,0.1,0.1];
% 
     f_presc=[A.*2.2/1e6;B.*2.2/1e6;C.*2.2/1e6;D.*2.2/1e6;E.*2.2/1e6].*1e3;
     f_presc([3:4;7:8]) = f_presc([7:8;3:4]); %prohozene osy => B<-->D
% 
   % x01=[6.15,-7.94,424.24,1.24.*pi/180,0.46.*pi/180,0.*pi/180];
   x01=x+[1,1,1,0.00001,0.00001,0.00001]';% [5,10,-3,0.1*pi/180,0.1*pi/180,0.1*pi/180]';
   
   [x,it]=fmin_sd(x01(1).*1e-3,x01(2).*1e-3,(x01(3)).*1e-3,x01(4),x01(5),x01(6),S.*1e-3,f_presc,v,n,L,1e-4);
   x(1:3)=x(1:3)*1e3;

    [f] = value_f(x(1).*1e-3,x(2).*1e-3,x(3).*1e-3,x(4),x(5),x(6),S.*1e-3,v,n,L);
    
%     inv_Trasnform_matrix = [0 1 0; -1 0 0; 0 0 1];
%     inv_Trasnform_matrix = [0 1 0; 1 0 0; 0 0 1];
%     inv_Trasnform_matrix_rotation = [0 -1 0; 1 0 0; 0 0 1];
    
    x=[inv(Trasnform_matrix)*x(1:3);inv(Trasnform_matrix_rotation)*x(4:6)];
    
    R = rotation_matrix(x(4),x(5),x(6));
    xx = x;
    xx(1:3) = x(1:3) - 1e3*L*R*v;
    
    RR = rotation_matrix(X(j,4),X(j,5),X(j,6));
    
    XX_temp = (X(j,1:3))';
    XX_temp(3) = min([XX_temp(3),0]);
    XX(j,1:3) = XX_temp' - (1e3*L*RR*v)';
    
    [xx';XX(j,:)]
    obraz = f.*1e3/2.2;
    
    for i=1:5
        obraz(2*i-1) = obraz(2*i-1) + 2592/2;
        obraz(2*i) = obraz(2*i) + 1944/2;
        % obraz
    end
    
%     for i=1:5
%         obraz(2*i-1) = obraz(2*i-1) + 2592/2;
%         obraz(2*i) = 1944-(obraz(2*i) + 1944/2); %osova soumernost podle osy y
%         % obraz
%     end
    center=[2592/2;1944/2];
    
    % AA=[1291;987];
    % DD=[598;502];
    % CC=[1782;289];
    % BB=[1992;1475];
    % EE=[804;1681];
    AA=aa(j,:)';
    BB=bb(j,:)';
    CC=cc(j,:)';
    DD=dd(j,:)';
    EE=ee(j,:)';
    
    fprintf('%i. konfigurace', j);
    % AAA=[AA obraz(1:2)]
    % BBB=[BB obraz(3:4)]
    % CCC=[CC obraz(5:6)]
    % DDD=[DD obraz(7:8)]
    % EEE=[EE obraz(9:10)]
    [AA' DD' CC' BB' EE';obraz']
    [AA' DD' CC' BB' EE']-[obraz']
    sum(([AA' DD' CC' BB' EE']-[obraz']).^2)
    max(abs([[AA' DD' CC' BB' EE']-[obraz']]))
    
    fprintf(fileID,'\n \n%i. konfigurace \n \n', j);
    fprintf(fileID,'\\begin{tabular}{|c|c|c|c|c|c|c|}\n');
    fprintf(fileID,'\\hline  && A & B & C & D & E \\\\ \\hline \n',AA(1),DD(1),CC(1),BB(1),EE(1));
    fprintf(fileID,'\\multirow{2}{*}{Elvac} & $x$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ \\\\ \n',AA(1),DD(1),CC(1),BB(1),EE(1));
    fprintf(fileID,'&$y$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ \\\\ \\hline \n',AA(2),DD(2),CC(2),BB(2),EE(2));
    fprintf(fileID,'\\multirow{2}{*}{Camera}& $x$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$  \\\\ \n',obraz(1),obraz(3),obraz(5),obraz(7),obraz(9));
    fprintf(fileID,'& $y$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$  \\\\ \\hline\n',obraz(2),obraz(4),obraz(6),obraz(8),obraz(10));

    eucDist1=sqrt(sum((AA-obraz(1:2)).^2));
    eucDist2=sqrt(sum((DD-obraz(3:4)).^2));
    eucDist3=sqrt(sum((CC-obraz(5:6)).^2));
    eucDist4=sqrt(sum((BB-obraz(7:8)).^2));
    eucDist5=sqrt(sum((EE-obraz(9:10)).^2));
    
    fprintf(fileID,'Euc. distance & & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$  \\\\ \\hline \n',eucDist1,eucDist2,eucDist3,eucDist4,eucDist5);
    fprintf(fileID,'\\end{tabular}');
    
    fprintf(fileID2,'\n \n%i. konfigurace \n \n', j);
    fprintf(fileID2,'\\begin{tabular}{|c|c|c|c|c|c|c|}\n');
    fprintf(fileID2,'\\hline & X & Y & Z & alpha & beta & gamma \\\\ \\hline \n');
    fprintf(fileID2,'{Matlab} & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ \\\\ \n',xx(1),xx(2),xx(3),xx(4)*180/pi,xx(5)*180/pi,xx(6)*180/pi);
    fprintf(fileID2,'{Elvac} & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ & $%5.1f$ \\\\ \\hline \n',XX(j,1),XX(j,2),XX(j,3),XX(j,4)*180/pi,XX(j,5)*180/pi,XX(j,6)*180/pi);
    fprintf(fileID2,'\\end{tabular}');
   
end

    fclose(fileID);
    fclose(fileID2);

%verze 3. 8. 2022

%zakladni jednotky v milimetrech

% images=readmatrix('body_29072022.xlsx');
% 
% camera_coordinates=readmatrix('Sou�adnice_29072022.xlsx');
% 
% %Prevod mezi globalnim souradnym systemem a souradnym systemem ELVACu, alpha=beta=0, gamma=-pi/2
% %osa x je naopak oto�en�!  m�me prohozeny osy x a y a kolem y rotujeme opa�n�!
% X=camera_coordinates;
% 
% X(:,4:6)=X(:,4:6).*pi/180;
% 
% 
% aa=images(:,1:2);
% bb=images(:,3:4);
% cc=images(:,5:6);
% dd=images(:,7:8);
% ee=images(:,9:10);
%     
% 
% %transformace Bail-Beremlijski syst�m -> ELVAC syst�m
% Trasnform_matrix = [0 -1 0; 1 0 0; 0 0 -1];
% Trasnform_matrix_rotation = [0 1 0; -1 0 0; 0 0 1];
% 
% %x=[0.0056;-0.0177;299.98;0.0005*pi/180;0.0034*pi/180;10.0005*pi/180];
% 
% 
% 
% 
% %kalibracni terc
% S=[0;0;0;0.05;0.05;0;0.05;-0.05;0;-0.05;-0.05;0;-0.05;0.05;0].*1e3;
% 
% v=[0;0;-1];
% n=[0;1;0];
% 
% L = 12.5.*1e-3;
% L = 9.5*1e-3;
% 
% 
% fileID = fopen('Table.txt','w');
% 
% for j=31:102
%        
%     xPresc=X(j,:)';
%     xPresc=[Trasnform_matrix*xPresc(1:3);Trasnform_matrix_rotation*xPresc(4:6)];
%         
%     %L mus� b�t upraveno dle z
%     L = min([309.5*1e-3 309.5*1e-3 + xPresc(3).*1e-3]);
%     xPresc(3) = max([xPresc(3),0]);
% 
% %     y=x; y(3)=L;
% %     [x_temp] = transformace(zeros(1,6),[zeros(3,1);x(4:6)]',y');
% %     x_temp(3)=x(3);
% %     
% %     x = x_temp;
%         
%     SS=[2592/2;1944/2];
%     
%     A=aa(j,:)'-SS;
%     B=bb(j,:)'-SS;
%     C=cc(j,:)'-SS;
%     D=dd(j,:)'-SS;
%     E=ee(j,:)'-SS;
%     x01=[0.1,0.1,0.1,0.1,0.1,0.1];
% 
%     f_presc=[A.*2.2/1e6;B.*2.2/1e6;C.*2.2/1e6;D.*2.2/1e6;E.*2.2/1e6].*1e3;
%     f_presc([3:4;7:8]) = f_presc([7:8;3:4]); %prohozene osy => B<-->D
% 
% 
%     [x,it]=fmin_sd(x01(1).*1e-3,x01(2).*1e-3,x01(3).*1e-3,x01(4),x01(5),x01(6),S.*1e-3,f_presc,v,n,L,1e-8); 
%     [f] = value_f(x(1).*1e-3,x(2).*1e-3,x(3).*1e-3,x(4),x(5),x(6),S.*1e-3,v,n,L);
%     
%     x(1:3)=x(1:3).*1e3;
% 
%     obraz = f.*1e3/2.2;
%     
%     for i=1:5
%         obraz(2*i-1) = obraz(2*i-1) + 2592/2;
%         obraz(2*i) = obraz(2*i) + 1944/2;
%         % obraz
%     end
%     center=[2592/2;1944/2];
%     
%     % AA=[1291;987];
%     % DD=[598;502];
%     % CC=[1782;289];
%     % BB=[1992;1475];
%     % EE=[804;1681];
%     AA=aa(j,:)';
%     BB=bb(j,:)';
%     CC=cc(j,:)';
%     DD=dd(j,:)';
%     EE=ee(j,:)';
%     
%     fprintf('%i. konfigurace', j);
%     % AAA=[AA obraz(1:2)]
%     % BBB=[BB obraz(3:4)]
%     % CCC=[CC obraz(5:6)]
%     % DDD=[DD obraz(7:8)]
%     % EEE=[EE obraz(9:10)]
%     [AA' DD' CC' BB' EE';obraz'];
%     [AA' DD' CC' BB' EE']-[obraz'];
%     sum(([AA' DD' CC' BB' EE']-[obraz']).^2);
%     max(abs([[AA' DD' CC' BB' EE']-[obraz']]));
%     
%     inv_Trasnform_matrix = [0 1 0; -1 0 0; 0 0 1];
%     inv_Trasnform_matrix = [0 1 0; 1 0 0; 0 0 1];
%     inv_Trasnform_matrix_rotation = [0 -1 0; 1 0 0; 0 0 1];
%     
%  %   x=[inv_Trasnform_matrix*x(1:3);inv_Trasnform_matrix_rotation*x(4:6)];
%     [xPresc';x']
