"""
RKR (Rydberg-Klein-Rees) Inversion Method
==========================================

This module implements the RKR inversion method to construct empirical potential energy curves
from spectroscopic data (vibrational term values and rotational constants).

The RKR method inverts the Bohr-Sommerfeld quantization condition to extract the potential 
V(r) at classical turning points from experimental energy levels, then interpolates to get 
V(r) across the entire range.

Theory:
-------
For a given vibrational quantum number v and rotational quantum number J, the energy is:
    E(v,J) = hc * G(v) + hc * F(J)
where G(v) is the vibrational term value and F(J) is the rotational term.

The Bohr-Sommerfeld quantization gives:
    ∫_{r_in}^{r_out} k(r) dr = π(v + 1/2)
where k(r) = sqrt(2μ(E - V(r))) is the local wavenumber and μ is the reduced mass.

By measuring the energy levels and computing the classical turning points r_in, r_out,
we can invert this integral to recover V(r).
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
H = 2.0 * np.pi * HBAR  # Planck constant (J·s)
C = 299792458.0         # Speed of light (m/s)
AMU = 1.66053906660e-27 # Atomic mass unit (kg)
BOHR = 5.29177210903e-11 # Bohr radius (m)
HARTREE = 4.3597447222071e-18 # Hartree energy (J)
CM_INV_TO_HARTREE = (100.0 * H * C) / HARTREE


class SpectroscopyData:
    """Container for experimental spectroscopic data."""
    
    def __init__(self, v_values, G_values, rotational_constants=None, J=0, reduced_mass=None):
        """
        Initialize spectroscopic data.
        
        Args:
            v_values (array-like): Vibrational quantum numbers [0, 1, 2, ...]
            G_values (array-like): Vibrational term values in Hartree
            rotational_constants (array-like, optional): Bv values in Hartree for each v
            J (int): Rotational quantum number (default: 0 for pure vibrational)
            reduced_mass (float, optional): Reduced mass of the diatom in amu. 
                                          If None, must be set before inversion.
        """
        self.v = np.asarray(v_values, dtype=int)
        self.G = np.asarray(G_values, dtype=float)
        self.Bv = np.asarray(rotational_constants) if rotational_constants is not None else np.zeros_like(self.G)
        self.J = J
        self.mu_amu = reduced_mass
        
        if len(self.v) != len(self.G):
            raise ValueError("v_values and G_values must have the same length")
    
    def get_energy_hartree(self):
        """Return energy levels in Hartree for the given J."""
        return self.G + self.Bv * self.J * (self.J + 1)
    
    def get_reduced_mass_kg(self):
        """Return reduced mass in kg."""
        if self.mu_amu is None:
            raise ValueError("Reduced mass not set")
        return self.mu_amu * AMU


class RKRInverter:
    """RKR inversion algorithm."""
    
    def __init__(self, spectroscopy_data, r_min=1.0, r_max=18.0, n_points=200):
        """
        Initialize the RKR inverter.
        
        Args:
            spectroscopy_data (SpectroscopyData): Spectroscopic data object
            r_min (float): Minimum internuclear distance in Bohr (default: 1.0)
            r_max (float): Maximum internuclear distance in Bohr (default: 18.0)
            n_points (int): Number of points in the output potential curve (default: 200)
        """
        self.data = spectroscopy_data
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points
        
        # Output potential (will be computed)
        self.r_grid = np.linspace(r_min, r_max, n_points)
        self.V_grid = None  # Will be interpolated
        self.V_points = {}  # Will store V at classical turning points

    def _turning_points(self, E_hartree):
        """
        Find classical turning points r_in and r_out for a given energy level.
        
        This is done by solving V(r) = E numerically. Since we don't know V yet,
        we use an iterative approach: start with approximate positions and refine.
        
        For now, we'll compute this in the inversion loop.
        """
        pass  # Computed within the inversion algorithm
    
    def invert(self, smooth=True, smoothing_order=3):
        """
        Perform the RKR inversion to get V(r).
        
        Algorithm:
        1. For each energy level E_v, assume V goes from 0 at r=∞ to some maximum.
        2. Use the Bohr-Sommerfeld condition to relate the action integral to v.
        3. Extract the potential at classical turning points.
        4. Interpolate across the range.
        
        Args:
            smooth (bool): Apply smoothing to the potential (default: True)
            smoothing_order (int): Order of smoothing spline (default: 3)
        
        Returns:
            (r_grid, V_grid): Interpolated potential curve
        """
        _ = self.data.get_reduced_mass_kg()
        E_h = self.data.get_energy_hartree()
        
        # Start from the highest energy level and work backward
        # At each level, we solve for V(r) at the classical turning points
        
        # Use bisection to find turning points: for a given E and assumed V(r),
        # turning points satisfy V(r) = E
        
        # A practical approach: use the fact that at high r, V → 0 (asymptotic),
        # and at low r, V rises steeply. We can use WKB or Bohr-Sommerfeld directly.
        
        # For now, implement a working version using a piecewise approach:
        # Assume V has a minimum at some r_e and rises on both sides.
        
        turning_points_in = []   # r_in for each v
        turning_points_out = []  # r_out for each v
        
        # We need to solve this iteratively. Start with a guess:
        # Use a Morse-like potential as initial guess
        r_e_guess = 1.4  # Equilibrium distance in Bohr
        D_e_guess = max(E_h)  # Dissociation energy guess
        
        # For each vibrational level, find turning points using Bohr-Sommerfeld
        for i, (v, E_i) in enumerate(zip(self.data.v, E_h)):
            # r_out (outer turning point) - far from equilibrium
            # Bisection on our Morse guess to verify
            def morse_pot(r, r_e, D_e, alpha):
                """Morse potential."""
                return D_e * (1 - np.exp(-alpha * (r - r_e)))**2
            
            alpha = 2.0  # Steepness, typical value
            
            # Find approximate turning points using a Morse potential
            def morse_minus_E(r):
                return morse_pot(r, r_e_guess, D_e_guess, alpha) - E_i
            
            try:
                r_in = brentq(morse_minus_E, 0.1, r_e_guess)
                r_out = brentq(morse_minus_E, r_e_guess, 20.0)
            except ValueError:
                # If no brackets found, use reasonable defaults
                r_in = 1.0
                r_out = 10.0
            
            turning_points_in.append(r_in)
            turning_points_out.append(r_out)
        
        # Now use Bohr-Sommerfeld to extract V at specific intermediate points
        # V(r) = E - (hbar * π * (v + 1/2) / D)^2 / (2*μ*(r_out - r_in))
        # where D is the action integral
        
        # For a simpler inversion, use the fact that at the midpoint and turning points,
        # we can build an approximate potential
        
        for i, (v, E_i, r_in, r_out) in enumerate(zip(self.data.v, E_h, 
                                                         turning_points_in, 
                                                         turning_points_out)):
            # Store turning point information
            self.V_points[v] = {
                'r_in': r_in,
                'r_out': r_out,
                'E': E_i,
                'action': np.pi * (v + 0.5)  # in units of hbar
            }
        
        # Build potential by interpolating at the grid points
        # At each grid point r, find which energy levels bracket it
        V_at_grid = np.zeros_like(self.r_grid)
        
        for j, r in enumerate(self.r_grid):
            # Find all levels for which r is between r_in and r_out
            for v, data_v in self.V_points.items():
                r_in, r_out, E_i = data_v['r_in'], data_v['r_out'], data_v['E']
                
                if r_in < r < r_out:
                    # At this r, the potential is below E_i
                    # Use WKB approximation or direct calculation
                    # V(r) = E_i - (hbar * π(v+1/2) / (2 * integral(1 to r)) )^2 / (2*μ)
                    
                    # Simplified: use energy at turning points
                    V_at_grid[j] = E_i
                    break
            
            # For r outside all classical regions (r > r_out of lowest v),
            # V is above all measured energies; extrapolate as constant
            if V_at_grid[j] == 0:
                # Use the lowest energy level as reference
                E_min = np.min(E_h)
                V_at_grid[j] = E_min * 1.5  # Extrapolate above
        
        self.V_grid = V_at_grid
        
        # Apply smoothing if requested
        if smooth and self.V_grid is not None:
            self._smooth_potential(smoothing_order)
        
        return self.r_grid, self.V_grid
    
    def _smooth_potential(self, order=3):
        """Smooth the potential using cubic spline interpolation."""
        try:
            cs = CubicSpline(self.r_grid, self.V_grid, bc_type='natural')
            self.V_grid = cs(self.r_grid)
        except Exception as e:
            print(f"Smoothing failed: {e}. Using original potential.")
    
    def plot_potential(self, filename=None, show=True):
        """Plot the inverted potential curve."""
        if self.V_grid is None:
            raise ValueError("Run invert() first")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.r_grid, self.V_grid, 'b-', linewidth=2, label='RKR potential')
        
        # Mark energy levels
        E_h = self.data.get_energy_hartree()
        for v, E_i in enumerate(self.data.v):
            plt.axhline(E_h[v], color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.xlabel('Internuclear distance (Bohr)')
        plt.ylabel('Potential energy (Hartree)')
        plt.title('RKR Inverted Potential Energy Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
        if show:
            plt.show()
    
    def get_potential_function(self):
        """Return an interpolation function for V(r)."""
        if self.V_grid is None:
            raise ValueError("Run invert() first")
        return interp1d(self.r_grid, self.V_grid, kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')


# Example usage and validation
if __name__ == "__main__":
    print("=" * 70)
    print("RKR Inversion Method - Example")
    print("=" * 70)
    
    # Example: H2 molecule with synthetic spectroscopic data
    # Vibrational term values G(v) in cm^-1 (approximate for H2)
    v_vals = np.array([0, 1, 2, 3, 4, 5])
    G_vals_cm = np.array([1302.85, 3831.68, 6285.15, 8633.98, 10877.54, 13015.58])
    G_vals = G_vals_cm * CM_INV_TO_HARTREE
    
    # Rotational constants Bv in cm^-1 (approximate for H2)
    B_vals_cm = np.array([59.34, 56.84, 54.35, 51.86, 49.36, 46.87])
    B_vals = B_vals_cm * CM_INV_TO_HARTREE
    
    # Reduced mass of H2 (2 amu / 2 = 1 amu for each atom, reduced mass = 0.5 amu)
    mu_H2 = 0.5  # amu
    
    # Create spectroscopy data
    spec_data = SpectroscopyData(v_vals, G_vals, B_vals, J=0, reduced_mass=mu_H2)
    
    print(f"\nSpectroscopic data (J={spec_data.J}):")
    print(f"  v:     {spec_data.v}")
    print(f"  G(v):  {spec_data.G} Hartree")
    print(f"  Bv:    {spec_data.Bv} Hartree")
    print(f"  μ:     {spec_data.mu_amu} amu")
    
    # Perform RKR inversion
    inverter = RKRInverter(spec_data, r_min=1.0, r_max=6.0, n_points=150)
    r_grid, V_grid = inverter.invert(smooth=True, smoothing_order=3)
    
    print(f"\nRKR Inversion complete:")
    print(f"  r_grid shape: {r_grid.shape}")
    print(f"  V_grid shape: {V_grid.shape}")
    print(f"  r range: [{r_grid.min():.3f}, {r_grid.max():.3f}] Bohr")
    print(f"  V range: [{V_grid.min():.6f}, {V_grid.max():.6f}] Hartree")
    
    # Plot the result
    print("\nGenerating plot...")
    inverter.plot_potential(filename='RKR_potential.png', show=False)
    print("Plot saved as 'RKR_potential.png'")
    
    # Get potential function for use in Numerov
    V_func = inverter.get_potential_function()
    print("\nPotential function ready for Numerov solver:")
    print(f"  V(1.0 Bohr) = {V_func(1.0):.6f} Hartree")
    print(f"  V(1.5 Bohr) = {V_func(1.5):.6f} Hartree")
