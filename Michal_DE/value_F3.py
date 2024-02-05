import jax.numpy as np


# Define the symbols
# alpha, beta, gama, X, Y, Z, L, v1, v2, v3, n1, n2, n3, ff, S_x, S_y, S_z = np.symbols(
#    'alpha beta gama X Y Z L v1 v2 v3 n1 n2 n3 ff S_x S_y S_z', real=True)


def fnc(x_input, f_obs, L):
    vec_XYZ = x_input[0:3]
    alpha = x_input[3]
    beta = x_input[4]
    gama = x_input[5]
    v = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ff = 0.008
    Z = vec_XYZ[2]

    # Rotation matrices Rx, Ry, Rz
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]], dtype=np.float32)
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float32)
    Rz = np.array([[np.cos(gama), -np.sin(gama), 0],
                    [np.sin(gama), np.cos(gama), 0], [0, 0, 1]], dtype=np.float32)

    R = Rx @ Ry @ Rz

    x = vec_XYZ - L * R @ v
    o = R @ v

    # Calculate 'a' using dot products
    a = np.sqrt(((-o[0] * Z) / (o[2]))**2 + (-(o[1] * Z) / (o[2]))**2 + Z * Z)

    # Calculate Q
    Q = a * o + vec_XYZ

    s_2 = R @ n
    s_1 = np.cross(o, s_2)
    s_1norm = np.sum(s_1**2)
    s_2norm = np.sum(s_2**2)

    M = np.array([[0, 0, 0],
                   [0.05, 0.05, 0],
                   [0.05, -0.05, 0],
                   [-0.05, -0.05, 0],
                   [-0.05, 0.05, 0]], dtype=np.float32)

    diff_sum = 0

    for i in range(5):
        # Substitute the symbols from M for S_x, S_y, S_z in f_1 and f_2
        Sip = M[i, :]
        Sip_proj = Sip + (o.dot(Q) - o.dot(Sip)) * (x - Sip) / (o.dot(x) - o.dot(Sip))
        # Calculate k_1 and k_2
        k_1 = (Sip_proj - Q).dot(s_1) / s_1norm
        k_2 = (Sip_proj - Q).dot(s_2) / s_2norm

        # Calculate image plane coordinates f_1 and f_2
        f_1_i = k_1 * ff / (a + L - ff)
        f_2_i = k_2 * ff / (a + L - ff)
        # Store the results in the matrix (also convert to mm)
        diff_sum += (f_obs[2 * i] - f_1_i * 1e3)**2
        diff_sum += (f_obs[2 * i + 1] - f_2_i * 1e3)**2

    return diff_sum
