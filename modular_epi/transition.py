"""Module defining transition and transmission links in a model."""

import numpy as np

class Parameter:
    """Defines a transition or transmission parameter and its name."""

    def __init__(self, name, val):
        """Constructor for Parameter.

        Args:
            name (str): the name of this parameter, e.g., "beta"
            val (float): the value for this parameter.
        """
        self.name = name
        self.value = val

    def __repr__(self):
        """Representation of a Parameter.

        Returns:
            str: representation of the parameter
        """
        return '{}: {}'.format(self.name, self.value)

class Transition:
    """Defines a compartment and a rate to exit to that compartment, to
       be used in the compartment that transitions to this
       OutCompartment.
    """

    def __init__(self, length, iters, compt=None, param=None, name=""):
        """Constructor for Transition.

        Args:
            length (int): the number of days to simulate.
            iters (int): the number of simulation iterations per day.
            compt (DiseaseCompartment): the disease compartment to
                transition to.
            param (Parameter): the Parameter for rate to transition to
                compt.
            name (str): the name of this out compartment. Should be
                formatted as {origin_compt} -> {out_compt} if simple
                transition, or {origin_compt} => {out_compt} if a
                transition at infection.
        """
        self.simulation_length = length
        self.intervals_per_day = iters
        self.incidence = np.zeros(iters * length)
        self.name = name
        self.compartment = compt
        self.parameter = param

    def reset(self, length=None, iters=None):
        """Reset this Transition.

        Args:
            length (int): new length of simulation, if specified.
            iters (int): new intervals per day, if specified.
        """
        if length is not None:
            self.simulation_length = length
        if iters is not None:
            self.intervals_per_day = iters
        self.incidence = np.zeros(iters * length)

    def add_value(self, val, t):
        """Add a value to the out compartment, used for transition.

        Args:
            val (float): the value to add to compt.
            t (int): the current time iteration to track incidence.
        """
        self.incidence[t] += val
        self.compartment.add_value(val)

    def __repr__(self):
        """Representation of this OutCompartment."""
        return self.name

class Transmission:
    """Defines a compartment and the rate of infection by that
       compartment, to be used in the SusceptibleCompartment that is
       infected by this compartment.
    """

    def __init__(self, N, compt=None, param=None, name=""):
        """Constructor for InfectiousCompartment.

        Args:
            compt (DiseaseCompartment): the disease compartment that is
                infectious.
            param (Parameter): the Parameter for rate of transmission to
                compt.
            name (str): the name of this InfectiousCompartment. Format
                is "{infectious_compt} *> {infected_comp}"
        """
        self.N = N
        self.compartment = compt
        self.parameter = param
        self.name = name

    @property
    def current_value(self):
        """Get current value of underlying compartment compt."""
        return self.compartment.current_value

    def __repr__(self):
        """Representation of this InfectiousCompartment."""
        return self.name