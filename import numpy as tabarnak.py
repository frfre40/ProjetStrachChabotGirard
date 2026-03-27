import pyscf as scf

###Initialisation de la molécule, de sa géométrie et de la base utilisée pour chaque atome
def constructionMolecule(element, basis, r):
    molecule = scf.gto.Mole()
    molecule.atom = f'''{element} 0 0 0; {element} {r} 0 0'''
    molecule.basis = basis

    return molecule

xavier = constructionMolecule('H', 'sto-3g', 1.0)







    