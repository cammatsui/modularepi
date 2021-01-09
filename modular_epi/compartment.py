"""Defines DiseaseCompartments and SusceptibleCompartments for
   compartmental models.
"""

import numpy as np

from transition import Transition, Transmission, Parameter

class DiseaseCompartment:
    """Superclass to define disease compartments in model."""

    def __init__(self, name, transitions_list=None, init_val=0,
                 length=1, iters=1, deterministic = True,
                 self_param=None):
        """The constructor for DiseaseCompartment class.

        Args:
            name (str): name of the compartment. should be unique.
            transitions_list (list): a list of Transitions, each linking
                this compartment and another with a parameter.
            init_val (float): the initial value of this compartment.
            length (int): the length of the simulation (in days).
            iters (int): the number of iterations to run per day
                for the model.
            self_param (Parameter): natural adding/subtracting to this
                compartment, such as from importation, birth, or death.
        """
        self.t = 0
        self.deterministic = deterministic
        if length <= 1:
            raise ValueError("Simulation length must be greater than 1!")
        self.simulation_length = length
        self.name = name
        self.intervals_per_day = iters
        self.initial_value = init_val
        self.total_time = length * iters
        self.values = np.zeros(self.total_time)
        self.values[0] = init_val
        self.transitions = transitions_list if transitions_list is not None \
            else []
        self.self_parameter = Parameter("", (np.zeros(self.total_time))) if \
            self_param is None else self_param

    def reset(self, length=None, iters=None):
        """Reset this DiseaseCompartment.

        Args:
            length (int): new simulation length, if specified.
            iters (int): new simulation intervals per day,
                if specified.
        """
        if length is None: length = self.simulation_length
        if iters is None: iters = self.intervals_per_day
        self.total_time = self.simulation_length * self.intervals_per_day
        self.values = np.zeros(self.total_time)
        self.t = 0
        self.values[0] = self.initial_value
        for transition in self.transitions:
            transition.reset(length=length, iters=iters)

    def add_transition(self, compt, param):
        """Add an transition from this compartment to compt via a
           Parameter.

        Args:
            compt (DiseaseCompartment): the compartment to transition
                to.
            param (Parameter): the rate at which to move to compt.

        Returns:
            Transition: created Transition object.
        """
        if len(param.value) != self.total_time:
            raise ValueError("Parameter value must be array with value "
                             "with length == total_time")
        transition = Transition(self.simulation_length,
            self.intervals_per_day, compt=compt, param=param,
            name='{} -> {}'.format(self.name, compt.name))
        self.transitions.append(transition)
        return transition

    def iterate_time(self):
        """Iterate the model's time value for this compartment. Should be
           done at the beginning of each model iteration for each
           compartment.
        """
        self.t += 1
        if self.t >= self.total_time:
            raise IndexError("Simulation ran for longer than initialized!")
        self.values[self.t] = self.values[self.t-1]

    def calculate_transition(self, transition):
        """Calculate the transition value for an out compartment.

        Args:
            transition (Transition): the Transition to calculate.

        Returns:
            float: the number of individuals transitioning from this
                compartment.
        """
        transition_num = self.current_value * (transition.parameter\
            .value[self.t-1] / self.intervals_per_day)
        if not self.deterministic:
            transition_num = np.random.poisson(transition_num)
        return transition_num

    def transition_all(self):
        """Transition out to all linked compartments and out of this
           compartment via self-parameter. Note: call only after
           iterating time for all compartments.
        """
        for transition in self.transitions:
            transition_val = self.calculate_transition(transition)
            self.add_value(-transition_val)
            transition.add_value(transition_val, self.t)
        self.add_value(-self.current_value * (self.self_parameter\
            .value[self.t-1] / self.intervals_per_day))

    def add_value(self, val):
        """Add a value to this compartment, typically if this compartment
           is an transitioned to by another compartment.

        Args:
            float: the value to add to this compartment.
        """
        self.values[self.t] += val

    @property
    def current_value(self):
        """Get the current value for calculating things like Force of
           Infection. Should be done after iteration.

        Returns:
            float: the value of this compartment in the last timestep.
        """
        current_val = self.values[self.t-1]
        return current_val

    @current_value.setter
    def current_value(self, val):
        """Set the current value of this compartment.

        Args:
            val (float): the new value of this compartment in this
                timestep.
        """
        self.values[self.t-1] = val

    def __repr__(self):
        """Representation of this DiseaseCompartment."""
        return self.name

class SusceptibleCompartment(DiseaseCompartment):
    """A class to define compartments with transitions from Infection
       (Susceptibles).
    """

    def __init__(self, name, N, transitions_list=None, init_val=0,
                 length=1, iters=1, transmissions=None,
                 transitions_at_infection=None,  deterministic=True,
                 self_param=None):
        """Constructor for SusceptibleCompartment. See
           DiseaseCompartment.

        Args:
            N (float): total number of individuals that are in this
                sub-population. Not used for transmission, but
                potentially necessary for Next Generation Matrix
                calculation.
            transmissions (list): a list of Transmissions, with
                parameters, that this compartment is infected through.
            transitions_at_infection (list): a list of Transitions that
                this compartment transitions through during infection.
        """
        super().__init__(name, transitions_list=transitions_list,
                     init_val=init_val, length=length, iters=iters,
                     deterministic=deterministic, self_param=self_param)
        self.transmissions = transmissions if transmissions is not None else []
        self.N = N
        self.transitions_at_infection = transitions_at_infection if \
            transitions_at_infection is not None else []

    def reset(self, length=None, iters=None):
        """Reset this SusceptibleCompartment.

        Args:
            length (int): new length for the simulation, if specified.
            iters (int): new intervals per day for simulation, if
                specified.
        """
        if length is None: length = self.simulation_length
        if iters is None: iters = self.intervals_per_day
        super().reset(length=length, iters=iters)
        for transition_at_infection in self.transitions_at_infection:
            transition_at_infection.reset(length=length, iters=iters)

    def add_transmission(self, compt, N, param):
        """
        Add an compartment to the list of compartments that infect
            this SusceptibleCompartment.

        Args:
            compt (DiseaseCompartment): the compartment to infect this
                SusceptibleCompartment via a Transmission.
            param (Parameter): the transmission rate.

        Returns:
            Transmission: created Transmission object.
        """
        transmission = Transmission(N, compt, param,
            name='{} *> {}'.format(self.name, compt.name))
        self.transmissions.append(transmission)
        return transmission

    def add_transition_at_infection(self, transition, prop=None):
        """Add a special Transition to a compartment at infection.

        Args:
            transition (DiseaseCompartment): the compartment to move to
                after exposure/infection.
            prop (Parameter): the proportion of infecteds that move
                through this specific Transition.

        Returns:
            Transition: the created Transition object which represents
                a transition at infection.
        """
        if prop is None: prop = Parameter("", np.ones(self.total_time))
        if len(prop.value) != self.total_time:
            raise ValueError("Parameter value must be array with value "
                             "with length == total_time")
        transition_at_infection = Transition(self.simulation_length,
            self.intervals_per_day, compt=transition, param=prop,
            name='{} => {}'.format(self.name, transition.name))
        self.transitions_at_infection.append(transition_at_infection)
        return transition_at_infection

    def calculate_infected(self, transmission):
        """Calculate the number infected via transmission Transmission.

        Args:
            transmission (Transmission): the transmission to calculate
                infections for.

        Returns:
            float: the number of susceptibles infected by
                this transmission in this timestep.
        """
        infected = ((transmission.parameter.value[self.t-1] 
                     / self.intervals_per_day)
            * self.current_value * transmission.current_value) / transmission.N
        if not self.deterministic: infected = np.random.poisson(infected)
        return infected

    def transition_all(self):
        """Calculate all transition values. Also calculates the force of
           infection on this SusceptibleCompartment and transitions
           infections.
        """
        self.transition_infections()
        super().transition_all()

    def transition_infections(self):
        """Transition all infections to the end node of each
           transmission.
        """
        sum_infected = 0
        for transmission in self.transmissions:
            infected = self.calculate_infected(transmission)
            sum_infected += infected
        self.add_value(-sum_infected)
        for trans_at_infection in self.transitions_at_infection:
            trans_at_infection.add_value(sum_infected
                * trans_at_infection.parameter.value[self.t-1], self.t)