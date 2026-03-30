import pyscf as scf
import numpy as np
from scipy.linalg import eigh

"""Ce module contient toutes les fonctions servant à trouver l'énergie totale de la molécule à l'étude en
   utilisant la méthode de Hartree-Fock. Celui-ci est organisé plusieurs sections: la première sert à bâtir les
   opérateurs utiles au calcul avec la méthode de Hartree-Fock, la seconde sert à diagonaliser l'opérateur de Fock
   pour trouver les énergies des orbitales moléculaires, et les coefficients d'occupation des orbitales.
   La troisième section contient les fonctions utiles pour faire la boucle SCF. Finalement, la quatrième section 
   comporte les fonctions pour faire le calcul de l'énergie totale de la molécule en additionnant les énergies
   calculées à partir de la méthode de Hartree-Fock à la répulsion nucléaire entre les atomes de la molécule."""

#############################################################################################
#############################################################################################
####Cette section contient les fonctions la création de l'objet représentant la molécule ####
## et les fonctions pour bâtir l'Hamiltonien, l'opérateur de Coulomb, d'échange et de Fock ##
#############################################################################################
#############################################################################################

#Initialisation de la molécule, de sa géométrie et de la base utilisée
def constructionMolecule(element, basis, r):
    """Construction de la molécule diatomique homonucléaire
    Args: 
        element (str): Symbole de l'élément chimique (ex: 'H' pour l'hydrogène)
        basis (str): Nom de la base utilisée (ex: 'sto-6g')
        r (float): Distance entre les deux atomes en unités de Bohr
    Returns: 
        molecule (pyscf.gto.Mole): Objet représentant la molécule construite"""
    molecule = scf.gto.Mole()
    molecule.atom = f'''{element} 0 0 0; {element} {r} 0 0'''
    molecule.basis = basis
    molecule.build()

    return molecule

H2 = constructionMolecule('H', 'sto-3g', 1.0)

#Calcul des intégrales pour construire le Hamiltonien core
def constructionHamiltonienCore(molecule):
    """Construction du Hamiltonien core pour une molécule donnée
    Args: 
        molecule (pyscf.gto.Mole): Objet représentant la molécule pour laquelle construire le Hamiltonien core
    Returns: 
        HamiltonienCore (numpy.ndarray): Matrice du Hamiltonien core"""
    T = molecule.intor("int1e_kin")   # kinetic integrals T_rs
    V = molecule.intor("int1e_nuc")   # nuclear attraction integrals V_rs
    return T + V

#Calcul des intégrales d'overlap
def constructionIntegralesOverlap(molecule):    
    """Construction de la matrice d'overlap S_rs pour une molécule donnée
    Args: 
        molecule (pyscf.gto.Mole): Objet représentant la molécule pour laquelle construire la matrice d'overlap
    Returns: 
        S (numpy.ndarray): Matrice d'overlap S_rs"""
    return molecule.intor("int1e_ovlp")  # overlap integrals S_rs


#Calcul des intégrales de répulsion électronique
def constructionIntegralesERI(molecule):
    """Construction des intégrales de répulsion électronique (rs|tu) pour une molécule donnée
    Args: 
        molecule (pyscf.gto.Mole): Objet représentant la molécule pour laquelle construire les intégrales de répulsion électronique
    Returns: 
        ERI (numpy.ndarray): Tableau des intégrales de répulsion électronique (rs|tu)"""
    return molecule.intor("int2e")      # two-electron repulsion integrals (rs|tu)


#Construction de la matrice de densité P_rs
def constructionMatriceDensite(coeffs, n_occ):
    """Construction de la matrice de densité P_rs pour RHF
    Args:
        coeffs (numpy.ndarray): Matrice des coefficients d'orbitales moléculaires
        n_occ (int): Nombre d'orbitales spatiales occupées (N_e/2)
    Returns:
        P (numpy.ndarray): Matrice de densité P_rs pour la molécule donnée"""
    coeffs_occ = coeffs[:, :n_occ]
    return 2.0 * coeffs_occ @ coeffs_occ.T
    

def OperateurFock(H, ERI, P):
    """Construction de l'opérateur de Fock F pour une molécule donnée
    Args: 
        H (numpy.ndarray): Matrice du Hamiltonien core
        ERI (numpy.ndarray): Tableau des intégrales de répulsion électronique (rs|tu)
        P (numpy.ndarray): Matrice de densité P_rs pour la molécule donnée. C'est une matrice carrée de taille n x n,
        où n est le nombre d'orbitales atomiques dans la base utilisée. Les éléments de P sont calculés à partir 
        des coefficients d'occupation des orbitales moléculaires et des coefficients de base.
    Returns: 
        F (numpy.ndarray): Matrice de l'opérateur de Fock F_rs pour la molécule donnée"""


    F = H.copy()
    for r in range(H.shape[0]):
        for s in range(H.shape[1]):
            for t in range(H.shape[0]):
                for u in range(H.shape[1]):
                    F[r, s] += P[t, u] * (ERI[r, s, t, u] - 0.5 * ERI[r, u, t, s])
    return F

#######################################################################################
#######################################################################################
##Cette section contient les fonctions pour la diagonalisation de l'opérateur de Fock##
#######################################################################################
#######################################################################################

def matriceOrthonormalisation(S, seuil=1e-10):
    """Construit la matrice X qui orthonormalise la base d'orbitales atomiques.
    Args:
        S (numpy.ndarray): Matrice d'overlap AO (symétrique définie positive)
        seuil (float): Valeur minimale acceptable pour les valeurs propres de S
    Returns:
        X (numpy.ndarray): Matrice d'orthonormalisation telle que X.T @ S @ X = I
    """
    valeursPropres, vecteursPropres = eigh(S)

    if np.any(valeursPropres <= seuil):
        raise ValueError(
            "La matrice d'overlap contient des valeurs propres trop petites. "
            "La base semble linéairement dépendante."
        )

    invRacine = np.diag(1.0 / np.sqrt(valeursPropres))
    X = vecteursPropres @ invRacine @ vecteursPropres.T
    return X


def transformationFock(F, X):
    """Calcule la transformation de F avant la diagonalisation de F
        Args: 
        F (numpy.ndarray): Matrice de l'opérateur de Fock F_rs pour la molécule donnée
        X (numpy.ndarray): Matrice d'orthonormalisation telle que X.T @ S @ X = I, où S
        est la matrice d'overlap.
        
        returns:
        Fprime (numpy.ndarray): Matrice de l'opérateur de Fock transformé F' = X.T @ F @ X, prête pour la diagonalisation"""
    return X.T @ F @ X

def diagonalisationFock(Fprime):
    """Diagonalise la matrice de l'opérateur de Fock transformé F'
    pour obtenir les énergies orbitalaires et les coefficients d'orbitales moléculaires
    Args: 
        Fprime (numpy.ndarray): Matrice de l'opérateur de Fock transformé F' = X.T @ F @ X, prête pour la diagonalisation
    Returns: 
        epsilon (numpy.ndarray): Vecteur des énergies orbitalaires (valeurs propres de F')
        Cprime (numpy.ndarray): Matrice des coefficients d'orbitales moléculaires dans la base orthonormale (vecteurs propres de F')"""
    valeursPropres, vecteursPropres = eigh(Fprime)
    epsilon = np.diag(valeursPropres)
    Cprime = vecteursPropres

    return epsilon, Cprime

def transformationCoefficients(Cprime, X):
    """Fait passer la matrice de coefficients d'orbitales moléculaires de la base orthonormale à la base d'orbitales atomiques
    
    Args:
        Cprime (numpy.ndarray): Matrice des coefficients d'orbitales moléculaires dans la base orthonormale (vecteurs propres de F')
        X (numpy.ndarray): Matrice d'orthonormalisation telle que X.T @ S @ X = I, où S est la matrice d'overlap.
    Returns:
        C (numpy.ndarray): Matrice des coefficients d'orbitales moléculaires dans la base d'orbitales atomiques"""
    return X @ Cprime

#################################################################
#################################################################
##Cette section contient les fonctions pour faire la boucle SCF##
#################################################################
#################################################################

def updateMatriceDensite(C, n_occ):
    """Met à jour la matrice de densité P_rs pour RHF
    Args:
        C (numpy.ndarray): Matrice des coefficients d'orbitales moléculaires dans la base AO
        n_occ (int): Nombre d'orbitales spatiales occupées (N_e/2)
    Returns:
        P (numpy.ndarray): Matrice de densité P_rs mise à jour pour la molécule donnée"""
    Cocc = C[:, :n_occ]
    return 2.0 * Cocc @ Cocc.T


def comparaisonDensite(P, Pmod, tolerance = 1e-8):
    """Critère d'arrêt pour la boucle SCF
    Args:
        P (numpy.ndarray): Matrice densité initiale obtenue à l'itération d'avant
        Pmod (numpy.ndarray): Matrice densité updatée avec la matrice C obtenue à l'itération actuelle.
        
    Returns: 
            arret (boolean): booleen qui dit au programme si le calcul a converge ou non """
    
    critere = np.allclose(P, Pmod, atol = tolerance)

    return critere