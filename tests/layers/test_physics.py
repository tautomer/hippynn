import unittest

import numpy as np
from sklearn.feature_selection import SelectFdr
import torch


class TestDipoleLayer(unittest.TestCase):

    # initialize the dipole layer
    from hippynn.layers.physics import Dipole

    dipole_layer = Dipole()
    # random number of molecules, atoms, and states
    n_mol, n_atoms, n_states = np.random.randint(2, 8, 3)
    # random initial positions and charges
    # set precision to numpy.float32
    positions = np.random.rand(n_mol * n_atoms, 3).astype("f")
    charges = np.random.rand(n_mol * n_atoms, n_states).astype("f")
    mol_index = np.repeat(np.arange(n_mol), n_atoms)

    def _dipole_numpy(self, charges: np.ndarray, positions: np.ndarray, mol_index: np.ndarray):
        """Same function as hippynn.layers.physics.Dipole.forward, but implemented in numpy.

        :param charges: charge array
        :type charges: np.ndarray
        :param positions: position array
        :type positions: np.ndarray
        :param mol_index: indices to sum over all atoms in one molecule
        :type mol_index: np.ndarray
        :return: calculated dipole
        :rtype: np.ndarray
        """
        n_states = charges.shape[1]
        if n_states > 1:
            positions = np.expand_dims(positions, 2)
            charges = np.expand_dims(charges, 1)
            dipole = np.zeros((self.n_mol, 3, n_states)).astype("f")
        else:
            dipole = np.zeros((self.n_mol, 3)).astype("f")
        dipole_elements = positions * charges
        np.add.at(dipole, mol_index, dipole_elements)
        return dipole

    def _assert_dipole(self, single=False, state=0):
        if single:
            charges = self.charges[:, state].reshape(-1, 1)
        else:
            charges = self.charges
        dipole_numpy = self._dipole_numpy(charges, self.positions, self.mol_index)
        dipole_torch = self.dipole_layer(
            torch.tensor(charges), torch.tensor(self.positions), torch.tensor(self.mol_index), self.n_mol
        ).numpy()
        self.assertTrue(np.array_equal(dipole_torch, dipole_numpy))
        return dipole_numpy, dipole_torch

    def test_dipole_layer(self):
        dipole_singles = np.empty((self.n_mol, 3, self.n_states))
        _, dipole_multi = self._assert_dipole()
        for i in range(self.n_states):
            _, tmp = self._assert_dipole(single=True, state=i)
            dipole_singles[:, :, i] = tmp
        self.assertTrue(np.array_equal(dipole_singles, dipole_multi))
