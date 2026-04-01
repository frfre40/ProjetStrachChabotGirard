import numpy as np
import matplotlib.pyplot as plt


###Si jamais ça plante, vérifier le signe de la partie potentielle dans la fonction Numerov, il se peut que j'ai fait une erreur de signe, et que ça soit pour ça que ça ne marche pas.


def Numerov(psi0, psi1, xi, xf, dx, potentiel):
    x = np.arange(xi, xf, dx, dtype = float)
    psi = np.zeros_like(x)
    psi[0] = psi0
    psi[1] = psi1

    longueurPsi = len(psi)
    for i in range(1, longueurPsi - 1):
        A = 2 * (1 - 5 * dx**2/12* potentiel[i]) * psi[i]
        B = (1 + dx**2/12*potentiel[i - 1]) * psi[i - 1]
        C = (1 + dx**2/12*potentiel[i + 1])

        psi[i+1] = (A - B) / C

    return psi


if __name__ == "__main__":
    # Simple validation case: for potentiel = 0, y'' = 0 and the solution is linear.
    xi, xf, dx = 0.0, 10.0, 0.05
    x = np.arange(xi, xf, dx, dtype=float)
    potentiel = np.zeros_like(x)

    psi0 = 0.0
    psi1 = dx

    psi_num = Numerov(psi0, psi1, xi, xf, dx, potentiel)
    psi_exact = x

    max_abs_err = np.max(np.abs(psi_num - psi_exact))
    print(f"Numerov self-test (V=0): max |error| = {max_abs_err:.3e}")

    if np.allclose(psi_num, psi_exact, rtol=1e-10, atol=1e-12):
        print("PASS")
    else:
        print("FAIL")

    # Harmonic oscillator (ground-state shape) validation:
    # psi_exact = exp(-x^2 / 2) satisfies psi'' = (x^2 - 1) * psi,
    # which corresponds to psi'' = -k(x) * psi with k(x) = 1 - x^2.
    xi_ho, xf_ho, dx_ho = -4.0, 4.0, 0.01
    x_ho = np.arange(xi_ho, xf_ho, dx_ho, dtype=float)
    potentiel_ho = 1.0 - x_ho**2

    psi_exact_ho = np.exp(-0.5 * x_ho**2)
    psi_num_ho = Numerov(psi_exact_ho[0], psi_exact_ho[1], xi_ho, xf_ho, dx_ho, potentiel_ho)

    max_abs_err_ho = np.max(np.abs(psi_num_ho - psi_exact_ho))
    print(f"Numerov harmonic-oscillator test: max |error| = {max_abs_err_ho:.3e}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_ho, psi_exact_ho, label="Exact: exp(-x^2/2)", linewidth=2)
    plt.plot(x_ho, psi_num_ho, "--", label="Numerov", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title("Numerov vs exact solution (harmonic oscillator test)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("numerov_harmonic_test.png", dpi=150)
    print("Plot saved as numerov_harmonic_test.png")

    abs_err_ho = np.abs(psi_num_ho - psi_exact_ho)
    plt.figure(figsize=(8, 4))
    plt.plot(x_ho, abs_err_ho, color="crimson", linewidth=1.8)
    plt.xlabel("x")
    plt.ylabel("|error|")
    plt.title("Pointwise absolute error: Numerov harmonic oscillator test")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("numerov_harmonic_error.png", dpi=150)
    print("Plot saved as numerov_harmonic_error.png")
    plt.show()


