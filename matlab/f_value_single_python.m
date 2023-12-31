function [res1, res2] = f_value_single_python(alpha, beta, gama, X, Y, Z, L, v1, v2, v3, n1, n2, n3, ff, S_x, S_y, S_z)
    x0 = sin(beta);
    x1 = cos(beta);
    x2 = cos(gama);
    x3 = v1.*x1.*x2;
    x4 = sin(gama);
    x5 = v2.*x1.*x4;
    x6 = -v3.*x0 + x3 + x5;
    x7 = sin(alpha);
    x8 = x2.*x7;
    x9 = cos(alpha);
    x10 = x4.*x9;
    x11 = -x0.*x10 + x8;
    x12 = -v2.*x11;
    x13 = x4.*x7;
    x14 = x2.*x9;
    x15 = v1.*(x0.*x14 + x13);
    x16 = v3.*x1;
    x17 = x16.*x9;
    x18 = x15 + x17;
    x19 = x12 + x18;
    x20 = n1.*x19 - n3.*x6;
    x21 = x16.*x7;
    x22 = v2.*(x0.*x13 + x14);
    x23 = v1.*(x0.*x8 - x10);
    x24 = x21 + x22 + x23;
    x25 = n1.*x24 - n2.*x6;
    x26 = -v2.*x11 + x18;
    x27 = -n3.*x24;
    x28 = -X;
    x29 = sqrt(v1.^2 + v2.^2 + v3.^2).*abs(Z/x26);
    x30 = x29.*x6;
    x31 = -L.*v3.*x0 + L.*x3 + L.*x5 + x28;
    x32 = -Z;
    x33 = L.*x12 + L.*x15 + L.*x17 + x32;
    x34 = -Y;
    x35 = L.*x21 + L.*x22 + L.*x23 + x34;
    x36 = S_x.*x6 + S_y.*x24 + S_z.*x19;
    x37 = x24.*x29;
    x38 = x19.*x29;
    x39 = (x19.*(Z + x38) + x24.*(Y + x37) - x36 + x6.*(X + x30))/(-x19.*x33 - x24.*x35 - x31.*x6 - x36);
    x40 = S_x + x28 - x30 + x39.*(-S_x - x31);
    x41 = S_y + x34 - x37 + x39.*(-S_y - x35);
    x42 = S_z + x32 - x38 + x39.*(-S_z - x33);
    x43 = ff/(L - ff + x29);
    res1 = x43.*(x20.*x41 - x25.*x42 + x40.*(-n2.*x19 - x27))/(x20.^2 + x25.^2 + (n2.*x26 + x27).^2);
    res2 = x43.*(n1.*x40 + n2.*x41 + n3.*x42)/(n1.^2 + n2.^2 + n3.^2);
    end
