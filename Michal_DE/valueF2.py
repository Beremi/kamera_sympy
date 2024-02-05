import jax.numpy as np


def val_f_unrolled(input_x, f_obs, L):
    X = input_x[0, :]
    Y = input_x[1, :]
    Z = input_x[2, :]
    alpha = input_x[3, :]
    beta = input_x[4, :]
    gama = input_x[5, :]
    f_obs1 = f_obs[0]
    f_obs2 = f_obs[1]
    f_obs3 = f_obs[2]
    f_obs4 = f_obs[3]
    f_obs5 = f_obs[4]
    f_obs6 = f_obs[5]
    f_obs7 = f_obs[6]
    f_obs8 = f_obs[7]
    f_obs9 = f_obs[8]
    f_obs10 = f_obs[9]
    x0 = np.cos(alpha)
    x1 = np.cos(beta)
    x2 = np.abs(Z * x0 / x1)
    x3 = x2 / x0**2
    x4 = (L + x3 - 0.008)**(-1.0)
    x5 = np.sin(gama)
    x6 = x1**2
    x7 = np.sin(alpha)
    x8 = np.cos(gama)
    x9 = np.sin(beta)
    x10 = x0 * x5
    x11 = x10 * x9 + x7 * x8
    x12 = -x0 * x8 + x5 * x7 * x9
    x13 = (x11**2 + x12**2 + x5**2 * x6)**(-1.0)
    x14 = L * x9 + X
    x15 = x14 * x9
    x16 = x0 * x1
    x17 = L * x16 + Z
    x18 = x16 * x17
    x19 = x1 * x7
    x20 = L * x19
    x21 = -Y + x20
    x22 = -x21
    x23 = -x1 * x22 * x7 + x15 + x18
    x24 = -1 / x23
    x25 = x3 * x9
    x26 = X - x25
    x27 = x26 * x9
    x28 = x1 * x2 / x0
    x29 = x16 * (Z - x28)
    x30 = Y + x19 * x3
    x31 = -x1 * x30 * x7 + x27 + x29
    x32 = -x31
    x33 = x24 * x32
    x34 = -X + x25
    x35 = x14 * x33 + x34
    x36 = x1 * x5
    x37 = -Z + x28
    x38 = x17 * x33 + x37
    x39 = -x12
    x40 = x22 * x24 * x32 - x30
    x41 = x14 + 0.05
    x42 = 0.05 * x9
    x43 = 0.05 * x19
    x44 = x42 + x43
    x45 = (-x23 - x44)**(-1.0)
    x46 = -x31 - x44
    x47 = x26 + 0.05
    x48 = x41 * x45 * x46 - x47
    x49 = x17 * x45 * x46 + x37
    x50 = -x21 - 0.05
    x51 = x30 - 0.05
    x52 = x45 * x46 * x50 - x51
    x53 = x14 - 0.05
    x54 = -x42 + x43
    x55 = (-x23 - x54)**(-1.0)
    x56 = -x31 - x54
    x57 = x55 * x56
    x58 = x34 + 0.05
    x59 = x53 * x57 + x58
    x60 = x17 * x57 + x37
    x61 = x50 * x55 * x56 - x51
    x62 = (-x15 - x18 + x19 * x22 + x44)**(-1.0)
    x63 = x19 * x30 - x27 - x29 + x44
    x64 = x62 * x63
    x65 = x53 * x64 + x58
    x66 = x17 * x64 + x37
    x67 = Y - x20 + 0.05
    x68 = x30 + 0.05
    x69 = x62 * x63 * x67 - x68
    x70 = x42 - x43
    x71 = (-x23 - x70)**(-1.0)
    x72 = -x31 - x70
    x73 = x41 * x71 * x72 - x47
    x74 = x17 * x71 * x72 + x37
    x75 = x67 * x71 * x72 - x68
    x76 = x10 * x6 + x11 * x9
    x77 = -x39 * x9 + x5 * x6 * x7
    x78 = x11 * x19 + x16 * x39
    x79 = (x76**2 + x77**2 + x78**2)**(-1.0)
    return (64.0 * (-0.125 * f_obs1 + x4 * x79 * (x35 * x78 + x38 * x77 + x40 * x76))**2 +
            64.0 * (-0.125 * f_obs10 + x13 * x4 * (x11 * x49 - x36 * x48 + x39 * x52))**2 +
            64.0 * (-0.125 * f_obs2 + x13 * x4 * (x11 * x38 - x35 * x36 + x39 * x40))**2 +
            64.0 * (-0.125 * f_obs3 + x4 * x79 * (x59 * x78 + x60 * x77 + x61 * x76))**2 +
            64.0 * (-0.125 * f_obs4 + x13 * x4 * (x11 * x60 - x36 * x59 + x39 * x61))**2 +
            64.0 * (-0.125 * f_obs5 + x4 * x79 * (x65 * x78 + x66 * x77 + x69 * x76))**2 +
            64.0 * (-0.125 * f_obs6 + x13 * x4 * (x11 * x66 - x36 * x65 + x39 * x69))**2 +
            64.0 * (-0.125 * f_obs7 + x4 * x79 * (x73 * x78 + x74 * x77 + x75 * x76))**2 +
            64.0 * (-0.125 * f_obs8 + x13 * x4 * (x11 * x74 - x36 * x73 + x39 * x75))**2 +
            64.0 * (-0.125 * f_obs9 + x4 * x79 * (x48 * x78 + x49 * x77 + x52 * x76))**2)
