import jax.numpy as jnp
from jax import grad, hessian, jit
import jax

jax.config.update("jax_enable_x64", True)


def value_F_raw(input_x, S, f_obs, v, n, L, ff=0.008):
    X = input_x[0]
    Y = input_x[1]
    Z = input_x[2]
    alpha = input_x[3]
    beta = input_x[4]
    gama = input_x[5]
    x0 = jnp.cos(alpha)
    x1 = jnp.cos(beta)
    x2 = x0 * x1
    x3 = v[2] * x2
    x4 = jnp.sin(alpha)
    x5 = jnp.cos(gama)
    x6 = x4 * x5
    x7 = jnp.sin(beta)
    x8 = jnp.sin(gama)
    x9 = x0 * x8
    x10 = x6 + x7 * x9
    x11 = v[1] * x10
    x12 = x4 * x8
    x13 = x0 * x5
    x14 = x12 - x13 * x7
    x15 = v[0] * x14
    x16 = x11 + x15 + x3
    x17 = jnp.sqrt(v[0]**2 + v[1]**2 + v[2]**2) * jnp.abs(Z / x16)
    x18 = 1 / (L - ff + x17)
    x19 = x1 * x5
    x20 = x1 * x8
    x21 = n[0] * x19 - n[1] * x20 + n[2] * x7
    x22 = n[0] * x14 + n[1] * x10 + n[2] * x2
    x23 = x1 * x4
    x24 = n[2] * x23
    x25 = x6 * x7 + x9
    x26 = n[0] * x25
    x27 = x12 * x7 - x13
    x28 = n[1] * x27 + x24 - x26
    x29 = 1 / (x21**2 + x22**2 + x28**2)
    x30 = -X
    x31 = v[2] * x7
    x32 = v[0] * x19
    x33 = -L * v[1] * x1 * x8 + L * x31 + L * x32 + x30
    x34 = v[2] * x23
    x35 = v[0] * x25
    x36 = -x27
    x37 = v[1] * x36
    x38 = -x34 + x35 + x37
    x39 = -Y
    x40 = -L * v[2] * x1 * x4 + L * x35 + L * x37 + x39
    x41 = -Z
    x42 = L * x11 + L * x15 + L * x3 + x41
    x43 = -v[1] * x20 + x31 + x32
    x44 = x16 * x42 + x33 * x43 + x38 * x40
    x45 = S[12] * x43 + S[13] * x38 + S[14] * x16
    x46 = x17 * x43
    x47 = x17 * x38
    x48 = x16 * x17
    x49 = -x16 * (Z + x48) - x38 * (Y + x47) - x43 * (X + x46)
    x50 = (-x45 - x49) / (-x44 - x45)
    x51 = x30 - x46
    x52 = S[12] + x50 * (-S[12] - x33) + x51
    x53 = x41 - x48
    x54 = S[14] + x50 * (-S[14] - x42) + x53
    x55 = n[1] * x36 - x24 + x26
    x56 = x39 - x47
    x57 = S[13] + x50 * (-S[13] - x40) + x56
    x58 = S[0] * x43 + S[1] * x38 + S[2] * x16
    x59 = (-x49 - x58) / (-x44 - x58)
    x60 = S[0] + x51 + x59 * (-S[0] - x33)
    x61 = S[2] + x53 + x59 * (-S[2] - x42)
    x62 = S[1] + x56 + x59 * (-S[1] - x40)
    x63 = S[3] * x43 + S[4] * x38 + S[5] * x16
    x64 = (-x49 - x63) / (-x44 - x63)
    x65 = S[3] + x51 + x64 * (-S[3] - x33)
    x66 = S[5] + x53 + x64 * (-S[5] - x42)
    x67 = S[4] + x56 + x64 * (-S[4] - x40)
    x68 = S[6] * x43 + S[7] * x38 + S[8] * x16
    x69 = (-x49 - x68) / (-x44 - x68)
    x70 = S[6] + x51 + x69 * (-S[6] - x33)
    x71 = S[8] + x53 + x69 * (-S[8] - x42)
    x72 = S[7] + x56 + x69 * (-S[7] - x40)
    x73 = S[9] * x43 + S[10] * x38 + S[11] * x16
    x74 = (-x49 - x73) / (-x44 - x73)
    x75 = S[9] + x51 + x74 * (-S[9] - x33)
    x76 = S[11] + x53 + x74 * (-S[11] - x42)
    x77 = S[10] + x56 + x74 * (-S[10] - x40)
    x78 = -x16 * x21 + x22 * x43
    x79 = v[1] * x27 + x34 - x35
    x80 = 1 / (x78**2 + (-x16 * x28 + x22 * x79)**2 + (-x21 * x79 + x28 * x43)**2)
    x81 = -x16 * x55 + x22 * x38
    x82 = -x78
    x83 = -x21 * x38 + x43 * x55
    result = ((-0.001 * f_obs[0] + ff * x18 * x80 * (x60 * x81 + x61 * x83 + x62 * x82))**2 +
              (-0.001 * f_obs[1] + ff * x18 * x29 * (x21 * x60 + x22 * x61 + x55 * x62))**2 +
              (-0.001 * f_obs[2] + ff * x18 * x80 * (x65 * x81 + x66 * x83 + x67 * x82))**2 +
              (-0.001 * f_obs[3] + ff * x18 * x29 * (x21 * x65 + x22 * x66 + x55 * x67))**2 +
              (-0.001 * f_obs[4] + ff * x18 * x80 * (x70 * x81 + x71 * x83 + x72 * x82))**2 +
              (-0.001 * f_obs[5] + ff * x18 * x29 * (x21 * x70 + x22 * x71 + x55 * x72))**2 +
              (-0.001 * f_obs[6] + ff * x18 * x80 * (x75 * x81 + x76 * x83 + x77 * x82))**2 +
              (-0.001 * f_obs[7] + ff * x18 * x29 * (x21 * x75 + x22 * x76 + x55 * x77))**2 +
              (-0.001 * f_obs[8] + ff * x18 * x80 * (x52 * x81 + x54 * x83 + x57 * x82))**2 +
              (-0.001 * f_obs[9] + ff * x18 * x29 * (x21 * x52 + x22 * x54 + x55 * x57))**2)
    return 1000000.0 * result


valF = jit(value_F_raw)
gradF = jit(grad(value_F_raw))
hessF = jit(hessian(value_F_raw))

_ = valF(jnp.zeros(6, dtype=jnp.float64), jnp.zeros(15, dtype=jnp.float64), jnp.zeros(10, dtype=jnp.float64),
         jnp.zeros(3, dtype=jnp.float64), jnp.zeros(3, dtype=jnp.float64), 1.0)
_ = gradF(jnp.zeros(6, dtype=jnp.float64), jnp.zeros(15, dtype=jnp.float64), jnp.zeros(10, dtype=jnp.float64),
          jnp.zeros(3, dtype=jnp.float64), jnp.zeros(3, dtype=jnp.float64), 1.0)
_ = hessF(jnp.zeros(6, dtype=jnp.float64), jnp.zeros(15, dtype=jnp.float64), jnp.zeros(10, dtype=jnp.float64),
          jnp.zeros(3, dtype=jnp.float64), jnp.zeros(3, dtype=jnp.float64), 1.0)
