import pyscf as scf
import numpy as np

###Initialisation de la molécule, de sa géométrie et de la base utilisée pour chaque atome
def constructionMolecule(element, basis, r):
    molecule = scf.gto.Mole()
    molecule.atom = f'''{element} 0 0 0; {element} {r} 0 0'''
    molecule.basis = basis
    molecule.build()

    return molecule

H2 = constructionMolecule('H', 'sto-6g', 1.0)

#Calcul des intégrales pour construire le Hamiltonien core
T = H2.intor("int1e_kin")   # kinetic integrals T_rs
V = H2.intor("int1e_nuc")   # nuclear attraction integrals V_rs
HamiltonienCore = T + V

#Calcul des intégrales d'overlap
S = H2.intor("int1e_ovlp")  # overlap integrals S_rs

#Calcul des intégrales à deux électrons
ERI = H2.intor("int2e")      # two-electron repulsion integrals (rs|tu)

print(np.shape(ERI))


#Calcul de l'opérateur de Hartree-Fock
def OperateurFock(H, ERI, P):
    F = H.copy()
    for r in range(H.shape[0]):
        for s in range(H.shape[1]):
            for t in range(H.shape[0]):
                for u in range(H.shape[1]):
                    F[r, s] += P[t, u] * (ERI[r, s, t, u] - 0.5 * ERI[r, u, t, s])
    return F

#
