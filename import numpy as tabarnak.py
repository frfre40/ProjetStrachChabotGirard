import numpy as np


class Electron:

    def __init__(self, orbitale, spin):
        self.orbitale = orbitale
        self.spin = spin


class Hamiltonien:

    def __init__(self, nbNoyaux, nbElectrons):
        self.nbNoyaux = nbNoyaux
        self.nbElectrons = nbElectrons

class FockOperator:

    def __init__(self, Hamiltonien, Coulomb, Exchange)