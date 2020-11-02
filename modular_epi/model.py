"""
Combines compartment and transition modules for a compartmental model
    object.
"""
from collections import defaultdict

import numpy as np

from compartment import DiseaseCompartment, SusceptibleCompartment
from transition import Parameter, Transition, Transmission

class CompartmentalModel:
    """
    An object to contain compartments, parameters, and methods for a 
        compartmental model.
    """

    def __init__(self, length, iters, compts=None):
        """
        Constructor for CompartmentalModel.

        Parameters:
            length (int): number of days for this model to run
            iters (int): number of iterations per day
            compts (list): list of compartments for this model.
        """
        compt_names = [c.name for c in compts] if compts is not None else []
        compts_list = compts if compts is not None else []
        # We use a dictionary for accessing compartments with strings.
        self.compartments = {n: c for n, c in zip(compt_names, compts_list)}
        self.simulation_length = length
        self.intervals_per_day = iters
        self.t = 1
        self.transitions = []
        self.transitions_at_infection = []
        self.transmissions = []

    def reset(self, length=None, iters=None):
        """
        Reset this model and all of its compartments.

        Parameters:
            length (int): new length for simulation, if specified.
            iters (int): new intervals per day for simulation, if
                specified.
        """
        for compartment in self.compartments:
            compartment.reset(length=length, iters = iters)
    
    def time_iterator(self):
        """
        Get an iterator to run simulations outside of the model's run 
            method.

        Returns:
            time_iterator (range): a range for running the model.
        """
        return range((self.simulation_length * self.intervals_per_day) - 1)
    
    def add_disease_compartment(self, name, init_val=0, self_param=("", 0)):
        """
        Add a (transition, can be infectious) DiseaseCompartment to this
            model.

        Parameters:
            name (str): the name of the DiseaseCompartment to be added.
            init_val (float): the initial number of individuals in the
                DiseaseCompartment to be added.
            self_param (tuple): the self parameter for this compartment 
                of form (name, value)
        """
        if name in self.compartments.keys():
            raise Warning(f"Compartment with name {name} already exists"
                          " in this model.")
        compt = DiseaseCompartment(name, init_val=init_val,
            length=self.simulation_length, iters=self.intervals_per_day,
            self_param=Parameter(self_param[0], self_param[1]))
        self.compartments[name] = compt
    
    def add_susceptible_compartment(self, name, init_val, N,
                                    self_param=("", 0)):
        """
        Add a SusceptibleCompartment to this model.

        Parameters:
            name (str): the name of the SusceptibleCompartment to be 
                added.
            N (float): the number of individuals in this sub-population.
                Used for setting S = N in the next generation matrix.
            init_val (float): the number of individuals initially
                in the SusceptibleCompartment to add.
            self_param (tuple): the self parameter for this compartment
                of form (name, value)
        """
        if name in self.compartments.keys():
            raise ValueError(f"Compartment with name {name} already"
                             " exists in this model.")
        compt = SusceptibleCompartment(name, N, init_val=init_val,
            length=self.simulation_length, iters=self.intervals_per_day,
            self_param=Parameter(self_param[0], self_param[1]))
        self.compartments[name] = compt

    def add_transmission(self, infected_compt_name, infectious_compt_name, N,
                         transmission_param):
        """
        Add transmission from compartment with name 
            "infectious_compt_name" to compartment with name 
            "infected_compt_name". 
        
        Parameters:
            infected_compt_name (str): the name of the compartment to be
                infected.
            infectious_compt_name (str): the name of the compartment to 
                be infectious.
            N (float): the number of individuals of this type.
            transmission_param (tuple): a parameter for the transmission
                rate between compartments of the form (name, value).
        """
        if not isinstance(self.compartments[infected_compt_name],
                          SusceptibleCompartment):
            raise ValueError("Infected compartment must be of type"
                             " SusceptibleCompartment")
        param = Parameter(transmission_param[0], transmission_param[1])
        new_transmission = self.compartments[infected_compt_name]\
            .add_transmission(self.compartments[infectious_compt_name], N, 
            param)
        self.transmissions.append(new_transmission)
        
    def add_transition(self, origin_compt_name, out_compt_name,
                       transition_param):
        """
        Add Transition from compartment with name "origin_compt_name" to
            compartment with name "out_compt_name".
        
        Parameters:
            origin_compt_name (str): name of the compartment that 
                individuals will transition from.
            out_compt_name (str): name of the compartment that
                individuals will transfer to.
            transition_param (tuple): a parameter for the rate of
                transition between compartments of the form
                (name, value).
        """
        for transition in self.compartments[origin_compt_name].transitions:
            if transition.name.split()[-1] == out_compt_name:
                raise Warning("There is already a transition between"
                              f" {origin_compt_name} and"
                              f" {out_compt_name}")
        param = Parameter(transition_param[0], transition_param[1])
        new_transition = self.compartments[origin_compt_name].add_transition(
            self.compartments[out_compt_name], param)
        self.transitions.append(new_transition)

    def add_transition_at_infection(self, susceptible_compt_name,
                                    exposed_compt_name,
                                    proportion_param=("", 1)):
        """
        Add transition from compartment with name 
            "susceptible_compt_name" to compartment with name 
            "exposed_compt_name" at infection.

        Parameters:
            susceptible_compt_name (str): name of the compartment of
                susceptibles.
            exposed_compt_name (str): name of the compartment that these
                susceptibles transition to.
            proportion_param (tuple): a parameter for the proportion
                of individuals that transition to the exposed compartment 
                of the form (name, value).
        """
        if not isinstance(self.compartments[susceptible_compt_name],
                          SusceptibleCompartment):
            raise ValueError("Infected compartment must be of type"
                             " SusceptibleCompartment.")
        param = Parameter(proportion_param[0], proportion_param[1])
        new_transition_at_infection = self.compartments[susceptible_compt_name]\
            .add_transition_at_infection(self.compartments[exposed_compt_name], 
            param)
        self.transitions_at_infection.append(new_transition_at_infection)
        
    def advance(self):
        """
        Advance the model simulation by one time-step.
        """
        for compt in self.compartments.values():
            compt.iterate_time()
        for compt in self.compartments.values():
            compt.transition_all()
        self.t += 1
    
    def run(self):
        """
        Run a simulation of the model.
        """
        for _ in self.time_iterator():
            self.advance()
    
    def get_current_metrics(self):
        """
        Get arrays for current compartment occupancy.

        Returns:
            current_metrics (dict): a dictionary with keys as 
                compartment names and values as arrays of compartment 
                occupancy over time.
        """
        current_metrics = {}
        total_time = self.simulation_length * self.intervals_per_day
        for name, compt in self.compartments.items():
            current_metrics[name] = compt.values[range(0, total_time,
                self.intervals_per_day)]
        return current_metrics
    
    def get_incidence(self):
        """
        Get arrays for incidence for each compartmental transition.

        Returns:
            incidence_metrics (dict): a dictionary with keys as 
                Transition names and values as incidence arrays.
        """
        incidence_metrics = {}

        def add_incidence(incid_dict, incid_key, incid):
            if incid_key in incid_dict.keys():
                incid_dict[incid_key] += incid  
            else:
                incid_dict[incid_key] = incid
            
        def get_daily_incidence(incid, length, iters):
            return incid.reshape(length, iters).sum(axis=1).transpose()

        for compt in self.compartments.values():
            for transition in compt.transitions:
                transitioned_to = transition.name.split()[-1]
                daily_incidence = get_daily_incidence(
                    transition.incidence,
                    self.simulation_length, self.intervals_per_day
                )
                add_incidence(incidence_metrics, transitioned_to,
                              daily_incidence)
            if isinstance(compt, SusceptibleCompartment):
                for trans_at_infection in compt.transitions_at_infection:
                    transitioned_to = trans_at_infection.name.split()[-1]
                    daily_incidence = get_daily_incidence(
                        trans_at_infection.incidence,
                        self.simulation_length, self.intervals_per_day
                    )
                    add_incidence(incidence_metrics, transitioned_to,
                                  daily_incidence)
        return incidence_metrics
    
    def get_cumulative(self):
        """
        Get arrays for cumulative incidence for each compartment.

        Returns:
            cumulative_metrics (dict): a dictionary with keys as
                DiseaseCompartment names and values as cumulative
                arrays.
        """
        incidence = self.get_incidence()
        cumulative_metrics = {}

        # TODO: Fix this!!
        for compt in incidence.keys():
            cumulative_ts = np.array([incidence[compt][:t].sum()
                for t in range(len(incidence[compt]))])
            if compt in cumulative_metrics.keys():
                cumulative_metrics[compt] += cumulative_ts
            else:
                cumulative_metrics[compt] = cumulative_ts
        return cumulative_metrics

    def get_transmissions(self):
        """
        Get transmissions for this model.

        Returns:
            transmission (list): a list of Transmissions for this model.
        """
        return self.transmissions
    
    def get_transitions(self):
        """
        Get transitions (not at infection) for this model.

        Returns:
            transitions (list): a list of Transitions for this model.
        """
        return self.transitions

    def get_transitions_at_infection(self):
        """
        Get transitions at infection for this model.

        Returns:
            transitions_at_infection (list): a list of transitions at
                infection for this model.
        """
        return self.transitions_at_infection
    
    def get_compartment(self, name):
        """
        Get a compartment object by its name.

        Returns:
            compt (DiseaseCompartment): the DiseaseCompartment with the 
                given name.
        """
        compt = self.compartments[name]
        return compt

    def fit_transmission_parameter(self, compartment_name, timeseries,
                                   timeseries_type='cumulative')
        """
        Fit the model's core transmission parameter to the
            given timeseries.
    
        Parameters:
            compartment_name (str): the name of the compartment
                to fit the data to.
            timeseries (np.array): the data to fit the compartment
                with name compartment_name to.
            timeseries_type (str): 'cumulative', 'current', or
                'incident', the type of timeseries to fit.
        
        Returns:
            transmission_param_val (float): the fitted value of the
                transmission parameter.
        """
        pass

    def fit_self_parameter(self):
        pass

    def fit_parameter(self):
        pass