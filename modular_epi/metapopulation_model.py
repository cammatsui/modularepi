"""Specifies a metapoopulation model object where multiple populations
   follow the same disease progression.
"""

import numpy as np

from compartment import DiseaseCompartment, SusceptibleCompartment
from transition import Parameter, Transition, Transmission
from model import CompartmentalModel

class MetapopulationModel(CompartmentalModel):
    """An object to contain a Metapopulation model."""

    def __init__(self, c, length, iters):
        """"Constructor for MetapopulationModel.

        Args:
            c (int): number of metapopulations.
            length (int): number of days for this model to run.
            iters (int): number of interations per day.
        """
        self.c = c
        self.simulation_length = length
        self.intervals_per_day = iters
        self.meta_compartments = []
        self.meta_susceptible_compartments = []
        self.meta_transitions = []
        self.meta_transitions_at_infection = []
        self.meta_transmissions = []
        super().__init__(compts=None, length=length, iters=iters)

    def _get_numbered_names(self, name):
        """Get names with subscript numbers for each metapopulation.

        Args:
            name (str): name to add subscript to.
        """
        return [f'{name}_{i}' for i in range(self.c)]

    def _parse_params(self, params, def_val):
        """Parse parameter of format (name, values) to format
           [(name_i, initial_value_i) for i in range(self.c)] or raise
           errors when necessary.

        Args:
            params (tuple): of format (name, values)
            def_val (float): the default value for the parameter.
        """
        if params is None:
            params = [("", def_val) for _ in range(self.c)]
        elif len(params[1]) != self.c:
            raise ValueError("Parameter values must have same length as "
                             "number of metapopulations.")
        else:
            param_names = self._get_numbered_names(params[0])
            param_vals = params[1]
            params = [(param_names[i], param_vals[i]) for i in range(self.c)]
        return params

    def add_disease_compartment(self, name, init_vals=None, self_params=None):
        """Add a disease compartment to this metapopulation model.

        Args:
            name (str): the broad name of this compartment, e.g.,
                "Susceptible."
            init_vals (list): a list of length self.c of initial values
                for each metapopulation comaprtment
            self_params (tuple): of format (name, values)
        """
        self.meta_compartments.append(name)
        names = self._get_numbered_names(name)
        init_vals = [0] * self.c if init_vals is None else init_vals
        self_params = self._parse_params(self_params, 0)
        for i in range(self.c):
            super().add_disease_compartment(names[i], init_vals[i],
                                            (self_params[i][0],
                                             self_params[i][1]))

    def add_susceptible_compartment(self, name, init_vals, Ns,
                                    self_params=None):
        """Add a susceptible compartment to this model.

        Args:
            name (str): the broad name of this susceptible compartment.
            init_vals (list): a list of floats, initial values for each
                metapopulation susceptible compartment.
            Ns (list): a list of floats, populations for each
                metapopulation.
            self_params (tuple): of form (name, vals)
        """

        self.meta_susceptible_compartments.append(name)
        names = self._get_numbered_names(name)
        if len(init_vals) != self.c:
            raise ValueError("Initial values must have same length as "
                             "number of metapopulations.")
        if len(Ns) != self.c:
            raise ValueError("Metapopulation sizes must have same length as "
                             "number of metapopulations.")
        self_params = self._parse_params(self_params, 0)
        for i in range(self.c):
            super().add_susceptible_compartment(names[i], init_vals[i], Ns[i],
                                                (self_params[i][0],
                                                 self_params[i][1]))

    def add_transition(self, origin_compt_name, out_compt_name,
                       transition_params):
        """Add a transition to this model.

        Args:
            origin_compt_name (str): the broad name of the compartment
                of origin.
            out_compt_name (str): the broad name of the destination
                comaprtment.
            transition_params (tuple): of form (name, vals)
        """
        self.meta_transitions.append("{} - {} -> {}".format(origin_compt_name,
            out_compt_name, transition_params[0]))
        transition_params = self._parse_params(transition_params, 0)
        origin_compt_names = self._get_numbered_names(origin_compt_name)
        out_compt_names = self._get_numbered_names(out_compt_name)
        for i in range(self.c):
            super().add_transition(origin_compt_names[i], out_compt_names[i],
                                   (transition_params[i][0],
                                    transition_params[i][1]))

    def add_transition_at_infection(self, susceptible_compt_name,
                                    exposed_compt_name,
                                    proportion_params=None):
        """Add a transition at infection to this model.

        Args:
            susceptible_compt_name (str): the broad name of the
                compartment to be infected.
            exposed_compt_name (str): the broad name of the compartment
                that infecteds transition to.
            proportion_params (tuple): of form (name, vals)
        """
        param_name = "" if proportion_params is None else proportion_params[0]
        self.meta_transitions_at_infection.append("{} - {} -> {}".format(
            susceptible_compt_name, param_name, exposed_compt_name))
        proportion_params = self._parse_params(proportion_params, 1)
        susceptible_compt_names = self.\
            _get_numbered_names(susceptible_compt_name)
        exposed_compt_names = self._get_numbered_names(exposed_compt_name)
        for i in range(self.c):
            super().add_transition_at_infection(susceptible_compt_names[i],
                                                exposed_compt_names[i],
                                                (proportion_params[i][0],
                                                 proportion_params[i][1]))

    def add_transmission(self, infected_compt_name, infectious_compt_name, Ns,
                         baseline_transmission_param, mixing_matrix):
        """Add a transmission to this model.

        Args:
            infected_compt_name (str): the broad name of the compartment
                to be infected.
            infectious_compt_name (str): the broad name of the
                compartment that infects the infected compartment.
            Ns (list): a list of length self.c of the number of
                individuals in each population.
            baseline_transmission_param (tuple): of form (name, vals)
            mixing_matrix (np.array): of shape (self.c, self.c).
                Describes mixing of metapopulations.
        """
        # Baseline transmission is relative infectousness of group j not
        # accounted for in matrix.
        infected_compt_names = self._get_numbered_names(infected_compt_name)
        infectious_compt_names = self._get_numbered_names(infectious_compt_name)
        param_name, param_vals = baseline_transmission_param
        param_names = self._get_numbered_names(param_name)
        for i in range(self.c):
            for j in range(self.c):
                if self.c == 1:
                    mat_val = mixing_matrix[0]
                else:
                    mat_val = mixing_matrix[i, j]
                super().add_transmission(
                    infected_compt_names[i],
                    infectious_compt_names[j],
                    Ns[j],
                    (param_names[j], param_vals[j] * mat_val)
                )

    def get_metrics(self, metric_type):
        """Get aggregated model outputs of a certain type.

        Args:
            metric_type (str): 'current', 'incidence', or 'cumulative'.

        Returns:
            dict: a dictionary with keys as compartment names and values
                as arrays of compartment occupancy over time.
        """
        if metric_type == 'current':
            metrics = super().get_current_metrics()
        elif metric_type == 'incidence':
            metrics = super().get_incidence()
        elif metric_type == 'cumulative':
            metrics = super().get_cumulative()
        else:
            raise ValueError("metric_type must be either 'current', 'incidence'"
                             ", or 'cumulative'!")
        metrics_agg = dict()
        meta_compartment_names = [name[:-2] for name in metrics.keys()]
        for meta_compartment_name in meta_compartment_names:
            metrics_agg[meta_compartment_name] = np.zeros(self.simulation_length)
            for name in self._get_numbered_names(meta_compartment_name):
                metrics_agg[meta_compartment_name] += metrics[name]
        return metrics_agg

    def get_current(self, agg=True):
        """Get arrays for current compartment occupancy.

        Args:
            agg (bool): whether or not to aggregate metacompartments.

        Returns:
            dict: a dictionary of daily current occupancy arrays with
                keys as compartment names.
        """
        if agg: return self.get_metrics('current')
        return super().get_current_metrics()

    def get_incidence(self, agg=True):
        """Get arrays for daily incidence to each compartment.

        Args:
            agg (bool): whether to aggregate metacompartments.

        Returns:
            dict: a dictionary of daily incidence arrays with keys as
                compartment names.
        """
        if agg: return self.get_metrics('incidence')
        return super().get_incidence()

    def get_cumulative(self, agg=True):
        """Get arrays for daily cumulative for each compartment.

        Args:
            agg (bool): whether to aggregate metacompartments.

        Returns:
            dict: a dictionary of daily cumulative arrays with keys as
                compartment names.
        """
        if agg: return self.get_metrics('cumulative')
        # TODO: check that this is right
        return super().get_incidence()