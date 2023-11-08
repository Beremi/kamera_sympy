function [t,it]=golden_section(S,f_presc,v,n,L,a,b,ddd,eps,x_t,y_t,z_t,alpha,beta,gamma)
% 1D minimization by golden section method
%
%input:     X,Y,Z,alpha,beta,gamma - skalary 
%               alpha,beta,gamma - in radians
%           S - vekt. 15x1 - souradnice bodu kalibracniho terce
%           f_presc - vekt. 10x1 - predepsany obraz
%           v - vekt. 3x1
%           n - vekt. 3x1
%           L - skalar - vzdalenost mezi objektivem a TCP bodem
%           a,b,eps - skalary
%           ddd - vekt. 6x1
%output:    t, it - skalary
%pomocne:   coeff - skalar
%           a0,b0,c0,d0,an,bn,cn,dn - skalary
%           fcn,fdn, fc,fd - skalary
%vse ve formatu double 

coeff=1/2 + sqrt(5)/2;

a0=a; b0=b;

d0=(b0-a0)/coeff + a0;
c0=a0+b0-d0;

it=0;

an=a0;
bn=b0;
cn=c0;
dn=d0;

t=(cn+dn)/2;

v0=v;
n0=n;

R=rotation_matrix(alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6));
n=R*n0;
fcn = valueF(x_t+cn*ddd(1),y_t+cn*ddd(2),z_t+cn*ddd(3),alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6),S,f_presc,v,n,L);

R=rotation_matrix(alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6));
n=R*n0;
fdn = valueF(x_t+dn*ddd(1),y_t+dn*ddd(2),z_t+dn*ddd(3),alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6),S,f_presc,v,n,L);

while bn-an > eps
   a=an; b=bn; 
   c=cn; d=dn;
   
   fc=fcn; fd=fdn;
   
   if fc < fd
      an=a; bn=d; dn=c; cn=an+bn-dn; t=dn;
      
      R=rotation_matrix(alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6));
      n=R*n0;
      fcn = valueF(x_t+cn*ddd(1),y_t+cn*ddd(2),z_t+cn*ddd(3),alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6),S,f_presc,v,n,L);
      fdn=fc;
   elseif fd < fc	
      an=c; bn=b; cn=d; dn=an+bn-cn; t=cn;
      fcn=fd; 
      
      R=rotation_matrix(alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6));
      n=R*n0;
      fdn=valueF(x_t+dn*ddd(1),y_t+dn*ddd(2),z_t+dn*ddd(3),alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6),S,f_presc,v,n,L);
   elseif fc == fd
      an=c; bn=d; dn=(bn-an)/coeff + an; cn=an+bn-dn;
      
      R=rotation_matrix(alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6));
      n=R*n0;
      fcn = valueF(x_t+cn*ddd(1),y_t+cn*ddd(2),z_t+cn*ddd(3),alpha+cn*ddd(4),beta+cn*ddd(5),gamma+cn*ddd(6),S,f_presc,v,n,L);
      
      R=rotation_matrix(alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6));
      n=R*n0;
      fdn = valueF(x_t+dn*ddd(1),y_t+dn*ddd(2),z_t+dn*ddd(3),alpha+dn*ddd(4),beta+dn*ddd(5),gamma+dn*ddd(6),S,f_presc,v,n,L);
      if fcn < fdn
         t=cn;
      else
         t=dn;
      end
   end
    
   it=it+1;
end

end

