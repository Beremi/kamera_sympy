import sympy as sp


def generate_valueF_sympy():
    # Define the symbols
    alpha, beta, gama, X, Y, Z, L, v1, v2, v3, n1, n2, n3, ff, S_x, S_y, S_z = sp.symbols(
        'alpha beta gama X Y Z L v1 v2 v3 n1 n2 n3 ff S_x S_y S_z', real=True)
    # Define the symbols for S
    S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15 = sp.symbols("S1:16", real=True)
    # observations for minsquare
    f_obs1, f_obs2, f_obs3, f_obs4, f_obs5, f_obs6, f_obs7, f_obs8, f_obs9, f_obs10 = sp.symbols('f_obs1:11', real=True)

    # Sip and TCP
    Sip = sp.Matrix([S_x, S_y, S_z])
    TCP = sp.Matrix([X, Y, Z])

    # Rotation matrices Rx, Ry, Rz
    Rx = sp.Matrix([[1, 0, 0], [0, sp.cos(alpha), -sp.sin(alpha)], [0, sp.sin(alpha), sp.cos(alpha)]])  # type: ignore
    Ry = sp.Matrix([[sp.cos(beta), 0, sp.sin(beta)], [0, 1, 0], [-sp.sin(beta), 0, sp.cos(beta)]])  # type: ignore
    Rz = sp.Matrix([[sp.cos(gama), -sp.sin(gama), 0], [sp.sin(gama), sp.cos(gama), 0], [0, 0, 1]])  # type: ignore
    R = sp.simplify(Rx * Ry * Rz)

    # Vector v and calculations for x and o
    v = sp.Matrix([v1, v2, v3])
    x = sp.Matrix([X, Y, Z]) - L * R * v
    o = R * v

    # Calculate 'a' using dot products
    a = sp.simplify(sp.sqrt(((-R.row(0).dot(v) * Z) / (R.row(2).dot(v)))**2 +
                            (-(R.row(1).dot(v) * Z) / (R.row(2).dot(v)))**2 + Z * Z))

    # Calculate Q
    Q = a * o + TCP

    # Project Sip onto the line defined by o and Q
    Sip_proj = Sip + (o.dot(Q) - o.dot(Sip)) * (x - Sip) / (o.dot(x) - o.dot(Sip))

    # s_2 vector and s_1 as the cross product of o and s_2
    s_2 = R @ sp.Matrix([n1, n2, n3])
    s_1 = o.cross(s_2)

    # Calculate k_1 and k_2
    k_1 = (Sip_proj - Q).dot(s_1) / s_1.norm()**2
    k_2 = (Sip_proj - Q).dot(s_2) / s_2.norm()**2

    # Q_t as a zero vector (for 2D image plane coordinates)
    Q_t = sp.Matrix([0, 0])

    # Calculate image plane coordinates f_1 and f_2
    f_1 = Q_t[0] + k_1 * ff / (a + L - ff)
    f_2 = Q_t[1] + k_2 * ff / (a + L - ff)

    M = sp.Matrix(5, 3, (S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15))

    # Define the matrix to store the results of f1 and f2 for all S
    results = sp.Matrix(10, 1, [0] * 10)

    for i in range(5):
        # Substitute the symbols from M for S_x, S_y, S_z in f_1 and f_2
        f1_i = f_1.subs({S_x: M[i, 0], S_y: M[i, 1], S_z: M[i, 2]})
        f2_i = f_2.subs({S_x: M[i, 0], S_y: M[i, 1], S_z: M[i, 2]})
        # Store the results in the matrix (also convert to mm)
        results[2 * i, 0] = f1_i * 1e3
        results[2 * i + 1, 0] = f2_i * 1e3

    f = sp.Matrix(10, 1, (f_obs1, f_obs2, f_obs3, f_obs4, f_obs5, f_obs6, f_obs7, f_obs8, f_obs9, f_obs10))

    # MinSquare functional
    diff = results - f
    F_obs = (diff[0] * diff[0] + diff[1] * diff[1] +
             diff[2] * diff[2] + diff[3] * diff[3] +
             diff[4] * diff[4] + diff[5] * diff[5] +
             diff[6] * diff[6] + diff[7] * diff[7] +
             diff[8] * diff[8] + diff[9] * diff[9])

    args = [X, Y, Z, alpha, beta, gama, L, v1, v2, v3, n1, n2, n3, ff,
            S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15,
            f_obs1, f_obs2, f_obs3, f_obs4, f_obs5, f_obs6, f_obs7, f_obs8, f_obs9, f_obs10]
    F_obs_lambd = sp.lambdify(args, F_obs, modules='jax', cse=True)

    def valueF_jax(x, v, n, L, ff, S, f):
        return F_obs_lambd(x[0], x[1], x[2], x[3], x[4], x[5], L, v[0], v[1], v[2], n[0], n[1], n[2], ff,
                           S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], S[9], S[10], S[11], S[12], S[13], S[14],
                           f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9])

    return valueF_jax
