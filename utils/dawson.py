import jax
import jax.numpy as jnp

# Chebyshev coefficients for approximating g(x)=dawsn(x)/x on x in [0,6],
# mapped via u = x/3 - 1 in [-1,1], with:
#   g(x) ≈ sum_{k=0}^{40} c[k] * T_k(u).
_DAWSN_G_CHEB_COEF = jnp.array(
    [
        0.2989781551634637,
        -0.46237666002333866,
        0.2482218345917861,
        -0.07447297292484911,
        -0.016626129599343956,
        0.03847645807169428,
        -0.02611650166905722,
        0.008675064213458414,
        0.001092974025998342,
        -0.0031177262056543325,
        0.0017668302541062424,
        -0.00032707564130682375,
        -0.00022851965294522612,
        0.00020524979916259844,
        -6.090212378731397e-05,
        -1.2064291186209925e-05,
        1.8477581326305537e-05,
        -6.419094285952632e-06,
        -5.863541627373464e-07,
        1.4381661383026553e-06,
        -5.030187392580721e-07,
        -4.273894015190616e-08,
        1.0042240537471124e-07,
        -3.105419048712196e-08,
        -4.358236043892631e-09,
        6.241926570564826e-09,
        -1.4871906181448028e-09,
        -4.0106785774415607e-10,
        3.348330415396632e-10,
        -4.9626657157497467e-11,
        -2.9102955756742883e-11,
        1.4857787692033075e-11,
        -5.855937315272723e-13,
        -1.646264293824482e-12,
        5.116032637015719e-13,
        5.704852010889178e-14,
        -7.245860311910573e-14,
        1.176705168767659e-14,
        4.9316278182827465e-15,
        -2.5517286276230627e-15,
        2.0560648706090988e-16,
    ],
    dtype=jnp.float64,
)


def _chebval(u: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate sum_{k=0}^N c[k] T_k(u) using Clenshaw recurrence.
    Works elementwise for any shape u.
    """
    b1 = jnp.zeros_like(u, dtype=c.dtype)
    b2 = jnp.zeros_like(u, dtype=c.dtype)
    # Static-length python loop is JIT-friendly (unrolled).
    for k in range(int(c.shape[0]) - 1, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]


@jax.jit
def dawsn_jax(x: jnp.ndarray, *, switch: float = 6.0, kasym: int = 20) -> jnp.ndarray:
    """
    Pure-JAX Dawson integral dawsn(x) for real x, compatible with jit/vmap.

    - Chebyshev approx on |x| <= switch
    - Asymptotic expansion on |x| > switch
    """
    x = jnp.asarray(x)
    dtype = x.dtype
    c = _DAWSN_G_CHEB_COEF.astype(dtype)

    ax = jnp.abs(x)

    # Chebyshev branch: dawsn(x) = x * g(|x|)
    u = ax / (switch / 2.0) - 1.0  # if switch=6 -> u = ax/3 - 1
    ghat = _chebval(u, c)
    daw_cheb = ax * ghat

    # Asymptotic branch for large ax:
    # dawsn(x) ~ sum_{n=0}^{kasym-1} c_n / x^(2n+1),
    # c_0=1/2, c_{n+1}=c_n * (2n+1)/2
    invx = 1.0 / (ax + 1e-300)
    invx2 = invx * invx
    coeff = 0.5
    term = invx
    s = coeff * term
    for n in range(1, kasym):
        coeff = coeff * (2.0 * n - 1.0) / 2.0
        term = term * invx2
        s = s + coeff * term
    daw_asym = s

    daw_pos = jnp.where(ax <= switch, daw_cheb, daw_asym)
    daw = jnp.sign(x) * daw_pos
    return jnp.where(x == 0.0, jnp.array(0.0, dtype=dtype), daw)


@jax.jit
def dawson_1f1(hyp_arg: jnp.ndarray) -> jnp.ndarray:
    """
    Stable 1F1(1; 3/2; hyp_arg) for hyp_arg <= 0:
        hyp_arg = -x (x>=0) -> 1F1(1;3/2;-x) = dawsn(sqrt(x))/sqrt(x)
    """
    hyp_arg = jnp.asarray(hyp_arg)
    x = jnp.maximum(-hyp_arg, 0.0)  # x>=0
    t = jnp.sqrt(x)
    daw = dawsn_jax(t)
    out = daw / (t + 1e-300)
    return jnp.where(x == 0.0, jnp.array(1.0, dtype=hyp_arg.dtype), out)
