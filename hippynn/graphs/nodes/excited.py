"""
Nodes for excited state modeling.
"""

from functools import partial
from typing import Tuple

import torch

from ...layers import excited as excited_layers
from .. import IdxType, find_unique_relative
from .base import AutoKw, ExpandParents, MultiNode, SingleNode
from .loss import _BaseCompareLoss
from .tags import AtomIndexer, Energies, HAtomRegressor, Network


class NACRNode(AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector multiplied by the energy difference
    between two states.
    """

    _input_names = "charges i", "charges j", "coordinates", "energy i", "energy j"
    _auto_module_class = excited_layers.NACR

    def __init__(
        self, name: str, parents: Tuple, module="auto", module_kwargs=None, **kwargs
    ):
        """Automatically build the node for calculating NACR * ΔE between two states i
        and j.

        :param name: name of the node
        :type name: str
        :param parents: parents of the NACR node in the sequence of (charges i, \
                charges j, positions, energy i, energy j)
        :type parents: Tuple
        :param module: _description_, defaults to "auto"
        :type module: str, optional
        :param module_kwargs: keyword arguments passed to the corresponding layer,
            defaults to None
        :type module_kwargs: dict, optional
        """

        self.module_kwargs = {}
        if module_kwargs is not None:
            self.module_kwargs.update(module_kwargs)
        charges1, charges2, positions, energy1, energy2 = parents
        positions.requires_grad = True
        self._index_state = IdxType.Molecules
        # self._index_state = positions._index_state
        parents = (
            charges1.main_output,
            charges2.main_output,
            positions,
            energy1.main_output,
            energy2.main_output,
        )
        super().__init__(name, parents, module=module, **kwargs)


class NACRMultiStateNode(AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector multiplied by the energy difference
    between all pairs of states.
    """

    _input_names = "charges", "coordinates", "energies"
    _auto_module_class = excited_layers.NACRMultiState

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        """Automatically build the node for calculating NACR * ΔE between all pairs of
        states.

        :param name: name of the node
        :type name: str
        :param parents: parents of the NACR node in the sequence of (charges, \
                positions, energies)
        :type parents: Tuple
        :param module: _description_, defaults to "auto"
        :type module: str, optional
        :param module_kwargs: keyword arguments passed to the corresponding layer,
            defaults to None
        :type module_kwargs: dict, optional
        """

        self.module_kwargs = {}
        if module_kwargs is not None:
            self.module_kwargs.update(module_kwargs)
        charges, positions, energies = parents
        positions.requires_grad = True
        self._index_state = IdxType.Molecules
        # self._index_state = positions._index_state
        parents = (
            charges.main_output,
            positions,
            energies.main_output,
        )
        super().__init__(name, parents, module=module, **kwargs)


class LocalEnergyNode(Energies, ExpandParents, HAtomRegressor, MultiNode):
    """
    Predict a localized energy, with contributions from implicitly computed atoms.
    """

    _input_names = (
        "hier_features",
        "mol_index",
        "atom index",
        "n_molecules",
        "n_atoms_max",
    )
    _output_names = (
        "mol_energy",
        "atom_energy",
        "atom_preenergy",
        "atom_probabilities",
        "atom_propensities",
    )
    _main_output = "mol_energy"
    _output_index_states = (
        IdxType.Molecules,
        IdxType.Atoms,
        IdxType.Atoms,
        IdxType.Atoms,
        IdxType.Atoms,
    )
    _auto_module_class = excited_layers.LocalEnergy

    @_parent_expander.match(Network)
    def expansion0(self, net, *, purpose, **kwargs):
        pdindexer = find_unique_relative(net, AtomIndexer, why_desc=purpose)
        return net, pdindexer

    @_parent_expander.match(Network, AtomIndexer)
    def expansion1(self, net, pdindexer, **kwargs):
        return (
            net,
            pdindexer.mol_index,
            pdindexer.atom_index,
            pdindexer.n_molecules,
            pdindexer.n_atoms_max,
        )

    _parent_expander.assertlen(5)

    def __init__(
        self, name, parents, first_is_interacting=False, module="auto", **kwargs
    ):
        parents = self.expand_parents(parents)
        self.module_kwargs = {"first_is_interacting": first_is_interacting}
        super().__init__(name, parents, module=module, **kwargs)

    def auto_module(self):
        network = find_unique_relative(self, Network).torch_module
        return self._auto_module_class(network.feature_sizes, **self.module_kwargs)


def _mae_with_phases(predict: torch.Tensor, true: torch.Tensor):
    """MAE with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MAE with phases
    :rtype: torch.Tensor
    """

    errors = torch.minimum(
        torch.linalg.norm((true - predict) / (torch.abs(true) + 1e-3), ord=1, dim=-1),
        torch.linalg.norm((true + predict) / (torch.abs(true) + 1e-3), ord=1, dim=-1),
    )
    return torch.sum(errors) / predict[..., 0].numel()


def _mse_with_phases(predict: torch.Tensor, true: torch.Tensor):
    """MSE with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MSE with phases
    :rtype: torch.Tensor
    """

    errors = torch.minimum(
        torch.linalg.norm((true - predict) / (torch.abs(true) + 1e-6), dim=-1),
        torch.linalg.norm((true + predict) / (torch.abs(true) + 1e-6), dim=-1),
    )
    return torch.sum(errors**2) / predict[..., 0].numel()


def _huber_loss_with_phases(
    predict: torch.Tensor, true: torch.Tensor, delta: torch.Tensor
):
    """Huber loss with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MSE with phases
    :rtype: torch.Tensor
    """

    # scaled_delta = torch.linalg.norm(true, ord=1, dim=-1) * delta
    minus = torch.linalg.norm(true - predict, ord=1, dim=-1)
    plus = torch.linalg.norm(true + predict, ord=1, dim=-1)
    diff = torch.minimum(minus, plus)

    quadratic = torch.minimum(diff, delta)
    # quadratic = torch.minimum(diff, scaled_delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    return torch.sum(loss) / predict[..., 0].numel()


def _huber_loss_with_phases(
    predict: torch.Tensor, true: torch.Tensor, delta: torch.Tensor
):
    """Huber loss with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MSE with phases
    :rtype: torch.Tensor
    """

    minus = torch.linalg.norm(true - predict, ord=1, dim=-1)
    plus = torch.linalg.norm(true + predict, ord=1, dim=-1)
    diff = torch.minimum(minus, plus)

    # Condition: Is the absolute true value smaller than delta?
    mse_mask = torch.linalg.norm(true, ord=1, dim=-1) < delta

    # MSE for true values smaller than delta
    mse_loss = 0.5 * diff**2

    # Linear and quadratic loss for true values greater than or equal to delta
    quadratic = torch.minimum(diff, delta)
    linear_loss = delta * (diff - quadratic)
    huber_loss = 0.5 * quadratic**2 + linear_loss

    # Combine both losses based on the condition
    loss = torch.where(mse_mask, mse_loss, huber_loss)

    # Return the mean loss over all elements
    return torch.sum(loss) / predict[..., 0].numel()


class MAEPhaseLoss(_BaseCompareLoss, op=_mae_with_phases):
    pass


class MSEPhaseLoss(_BaseCompareLoss, op=_mse_with_phases):
    pass


class HuberPhaseLoss(_BaseCompareLoss):

    def __init_subclass__(cls, delta=None):
        if delta is None:
            raise ValueError("`delta` must be given for a Huber loss function")
        huber = partial(_huber_loss_with_phases, delta=delta)
        huber.__name__ = "_huber_loss_with_phases"
        super().__init_subclass__(op=huber)
