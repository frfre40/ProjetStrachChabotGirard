import pyscf as scf
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

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
    epsilon = valeursPropres
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


def comparaisonDensite(P, Pmod, tolerance = 1e-10):
    """Critère d'arrêt pour la boucle SCF
    Args:
        P (numpy.ndarray): Matrice densité initiale obtenue à l'itération d'avant
        Pmod (numpy.ndarray): Matrice densité updatée avec la matrice C obtenue à l'itération actuelle.
        
    Returns: 
            arret (boolean): booleen qui dit au programme si le calcul a converge ou non """
    
    critere = np.allclose(P, Pmod, atol = tolerance, rtol=0.0)

    return critere



########################################################################
########################################################################
##Cette section contient la fonction pour faire le calcul Hartree-Fock##
########################################################################
########################################################################

def HartreeFock(element, basisSet, r):

    #1- initialisation de la molécule
    molecule = constructionMolecule(element, basisSet, r)
    n_occ = molecule.nelectron // 2

    #2- Construction du Hamiltonien
    Hcore = constructionHamiltonienCore(molecule)

    #3- Construction de l'opérateur de Coulomb et de l'opérateur d'échange 
    #(l'opérateur d'échange est construit à partir de l'opérateur de Coulomb, il n'est pas nécessaire de le construire séparément)
    ERI = constructionIntegralesERI(molecule)

    #4- Construction de la matrice d'overlap
    S = constructionIntegralesOverlap(molecule)

    #5- Construction la matrice de densité initiale (on peut choisir n'importe quelle matrice de densité, ici on choisit la matrice de densité nulle)
    #Ce guess initial est basé sur une suggestion faite dans le chapitre 13.16 de Quantum Chemistry par Levine
    n_orbitals = Hcore.shape[0]
    P = np.zeros((n_orbitals, n_orbitals))

    #6- Construisage de la matrice d'orthonormalisation
    X = matriceOrthonormalisation(S)

    #7- Construisage du premier guess pour Fock
    F = OperateurFock(Hcore, ERI, P)

    #8- Transformation de F avant la diagonalisation
    Fprime = transformationFock(F, X)

    #9- Diagonalisation de Fprime les valeurs propres et les vecteurs propres dans la base orthonormale, aka les énergies
    #et les coefficients d'orbitales moléculaires dans la base orthonormale
    epsilon, Cprime = diagonalisationFock(Fprime)

    #10-Pitchage des coefficients de Cprime de la base orthonormale à la base initiale (celle d'orbitales atomiques)
    C = transformationCoefficients(Cprime, X)

    #11- Updatage de la matrice de densité P à partir de la matrice de coefficients C obtenue à l'itération actuelle
    Pupdate = updateMatriceDensite(C, n_occ=n_occ)


    #12- SCF BABYYYYYYYYYY
    #P de l'itération initiale est comparée avec Pupdate de l'itération actuelle.
    #Si les deux matrices de densité sont égales (à une tolérance près), alors le calcul a convergé et la boucle s'arrête.
    #Sinon, on met P à jour avec Pupdate, et on recommence le calcul à partir de l'étape 

    compteur = 1


    while not comparaisonDensite(P, Pupdate):
        print(f"SCF iteration {compteur} - max difference in density matrix: {np.max(np.abs(P - Pupdate)):.2e}")
        P = Pupdate.copy()
        F = OperateurFock(Hcore, ERI, P)
        Fprime = transformationFock(F, X)
        epsilon, Cprime = diagonalisationFock(Fprime)
        C = transformationCoefficients(Cprime, X)
        Pupdate = updateMatriceDensite(C, n_occ=n_occ)
        compteur += 1
        if compteur > 400:
            break

    #Calcul de l'énergie trouvée avec Hartree-Fock. Cette équation est l'équation (13.168) de la 
    #cinquième édition de Quantum Chemistry par Levine
    energieMatrice = 0.5 * P @ (F + Hcore)
    energieElectroniqueHF = np.trace(energieMatrice)
    energieHF = energieElectroniqueHF + molecule.energy_nuc()

    return energieHF


if __name__ == "__main__":

    ###Test avec 3 basis sets différents sur plusieurs distances interatomiques
    liste1 = []
    liste2 = []
    liste3 = []
    liste4 = []
    distances = np.linspace(0.3, 3.0, 30)
    for r in distances:
        #Calcul avec fonction maison
        liste1.append(HartreeFock('H', 'sto-6g', r))
        liste2.append(HartreeFock('H', 'sto-3g', r))
        liste4.append(HartreeFock('H', 'def2-TZVP', r))
        #Calcul avec PySCF pour vérifier les résultats de la fonction maison
        molecule = constructionMolecule('H', 'sto-3g', r)
        mf = scf.scf.RHF(molecule)
        liste3.append(mf.kernel())

    liste1_raw = np.array(liste1)
    liste2_raw = np.array(liste2)
    liste3_raw = np.array(liste3)
    liste4_raw = np.array(liste4)

    min1 = np.min(liste1_raw)
    min2 = np.min(liste2_raw)
    min3 = np.min(liste3_raw)
    min4 = np.min(liste4_raw)

    liste1 = liste1_raw - min1
    liste2 = liste2_raw - min2
    liste3 = liste3_raw - min3
    liste4 = liste4_raw - min4

    def Morse(r, De, a, re):
        """Fonction de Morse pour faire un fit des données obtenues avec la fonction maison
        Args:
            r (numpy.ndarray): Tableau des distances interatomiques
            De (float): Profondeur du puits de potentiel
            a (float): Paramètre de largeur du puits de potentiel
            re (float): Distance d'équilibre entre les atomes
        Returns:
            V (numpy.ndarray): Tableau des énergies potentielles calculées à partir de la fonction de Morse pour les distances r"""
        return De * (1 - np.exp(-a * (r - re)))**2

    def MorseAvecOffset(r, De, a, re, Einf):
        """Potentiel de Morse avec offset: E(re)=Einf-De et E(r->inf)=Einf."""
        return Einf + De * (1 - np.exp(-a * (r - re)))**2 - De

    params, _ = curve_fit(
        MorseAvecOffset,
        distances,
        liste3_raw,
        p0=[0.1745, 1.04, 0.74, -1.0],
        bounds=([0.0, 0.0, 0.2, -2.0], [2.0, 10.0, 2.0, 0.0]),
        maxfev=10000,
    )
    De_fit, a_fit, re_fit, Einf_fit = params

    print("\nParametres Morse ajustes sur liste3 (PySCF STO-3G):")
    print(f"De   = {De_fit:.8f} Eh")
    print(f"a    = {a_fit:.8f} Bohr^-1")
    print(f"re   = {re_fit:.8f} Angstrom")
    print(f"Einf = {Einf_fit:.8f} Eh")
    print(f"Emin = {Einf_fit - De_fit:.8f} Eh")
        
    x = np.linspace(0.01, 6.0, 100)
    De = 0.1745
    a = 1.04
    re = 0.74
    morse_fit_shifted = MorseAvecOffset(x, De_fit, a_fit, re_fit, Einf_fit) - min3

    plt.plot(distances, liste1, marker='o', label='STO-6G (custom)')
    plt.plot(distances, liste2, marker='o', label='STO-3G (custom)')
    plt.plot(distances, liste4, marker='o', label='def2-TZVP (custom)')
    plt.plot(distances, liste3, marker='o', linestyle='--', label='STO-3G (PySCF)')
    #La fonction graphée avec Morse en utilisant les données expérimentales ne fonctionne pas
    #(elle ne monte pas violemment près de l'origine)
    plt.plot(x, morse_fit_shifted, linestyle='-', label='Morse fit (liste3)')
    plt.axhline(De, color='black', linestyle='--', linewidth=1.0)
    plt.xlabel('Distance interatomique (Bohr)')
    plt.ylabel('Energie totale (Hartree)')
    plt.title('Energie totale de H2 en fonction de la distance interatomique')
    plt.grid()
    plt.legend()
    plt.show()