"""Module to create a next generation matrix for a model to calculate
   R0.
"""

import numpy as np

class NGM:
    """Defines the Next Generation Matrix (NGM) for a given
       CompartmentalModel. Used to calculate R0.
    """

    def __init__(self, model):
        """Constructor for NGM.

        Args:
            model (CompartmentalModel): a model to construct the NGM for
        """
        self.model = model
        self.compartments = self._get_ngm_compartments()
        self.matrix_indices = {n: c for n, c in enumerate(self.compartments)}
        self.matrix_name_to_index = {c.name: n for n,
            c in enumerate(self.compartments)}

    def _get_ngm_compartments(self):
        """Get model compartments used in the NGM.

        Returns:
            list: a list of the DiseaseCompartments to be used in this
                NGM.
        """
        ngm_compts = set()
        compt_names = set()
        for c in self.model.transitions_at_infection:
            compt_names.add(c.name.split()[-1])
        for c in self.model.transmissions:
            compt_names.add(c.name.split()[-1])
        compt_names = sorted(list(compt_names))
        ngm_compts = [self.model.get_compartment(c) for c in compt_names]
        return ngm_compts

    def _get_transitions_at_infection_on(self, compt):
        """Get a list of Transitions to compartments at infection on
           the DiseaseCompartment compt for this model.

        Args:
            compt (DiseaseCompartment): the compartment of interest.

        Returns:
            list: a list of Transitions that give compartments at
                infection on compt.
        """
        transitions_at_infection = []
        for transition_at_infection in \
                self.model.transitions_at_infection:
            if compt.name == transition_at_infection.name.split()[-1]:
                transitions_at_infection.append(transition_at_infection)
        return transitions_at_infection

    def _get_transmissions_on(self, compt):
        """Get a list of Transmission that infect this compt, where
           compt is a SusceptibleCompartment.

        Args:
            compt (SusceptibleCompartment): the SusceptibleCompartment
                of interest.

        Returns:
            list: a list of Transmissions that infect compt.
        """
        compts = []
        for transmission in self.model.transmissions:
            if compt.name == transmission.name.split()[0]:
                compts.append(transmission)
        return compts

    def _get_transmission_entry(self, infected, infecting):
        """Get a transmission matrix entry for an infected compartment and
           infecting compartment.

        Args:
            infected (DiseaseCompartment): the ith compartment in matrix,
                infected by infecting.
            infecting (DiseaseCompartment): the jth compartment in
                matrix.

        Returns:
            float: the transmission matrix entry.
        """
        entry = 0
        transitions_at_infection_on = self.\
            _get_transitions_at_infection_on(infected)
        if not transitions_at_infection_on: return entry
        # Get proportion parameters.
        ps = [c.parameter.get_initial_value() for c in 
              transitions_at_infection_on]
        # Get susceptible compartments that feed into infected.
        transitions_at_infection_on_names = [c.name.split()[0] for c in
            transitions_at_infection_on]
        susceptible_compts = [self.model.get_compartment(n) for n in
            transitions_at_infection_on_names]
        # Get infectious compartments for each susceptible_compt that
        # have "infecting" as the infectious part.
        for susceptible_compt, p in zip(susceptible_compts, ps):
            S_n = susceptible_compt.N
            transmissions_on = self.\
                _get_transmissions_on(susceptible_compt)
            # Add FOI contribution to entry.
            for transmission_on in transmissions_on:
                if transmission_on.name.split()[-1] == infecting.name:
                    entry += (transmission_on.parameter.get_initial_value()
                        * (S_n / transmission_on.N) * p)

        return entry

    def construct_transmission_matrix(self):
        """Create transmission matrix (T) to create the NGM
           with large domain K_L.

        Returns:
            np.array: the transmission matrix.
        """
        transmission = np.zeros((len(self.compartments),
                                 len(self.compartments)))
        # compt_1 is infected by compt_2.
        for compt_1 in self.compartments:
            for compt_2 in self.compartments:
                i = self.matrix_name_to_index[compt_2.name]
                j = self.matrix_name_to_index[compt_1.name]
                transmission[j, i] = self.\
                    _get_transmission_entry(compt_1, compt_2)
        return transmission

    def _get_between_compartment(self, origin_compt, trans_compt):
        """Get Transition that transition to trans_compt from
           origin_compt.

        Args:
            origin_compt (DiseaseCompartment): the compartment to get
                the Transition coming from.
            out_compt (DiseaseCompartment): the compartment to get the
                Transition going to.

        Returns:
            None or OutCompartment: the OutCompartment between
                origin_compt and out_compt
        """
        for out_compt in self.model.transitions:
            names = out_compt.name.split()
            if names[0] == origin_compt.name and names[-1] == trans_compt.name:
                return out_compt
        return None

    def construct_transition_matrix(self):
        """Create transition matrix (sigma) to create the NGM
           with large domain K_L.

        Returns:
            np.array: the transition matrix.
        """
        transition = np.zeros((len(self.compartments), len(self.compartments)))
        # Fill out main diagonal.
        for i in range(len(self.compartments)):
            cur_compt = self.matrix_indices[i]
            entry = cur_compt.self_parameter.get_initial_value()
            for transition_link in cur_compt.transitions:
                entry += transition_link.parameter.get_initial_value()
            transition[i, i] = -entry
        # Fill out rest of matrix.
        for compt_1 in self.compartments:
            for compt_2 in self.compartments:
                btwn_compt = self._get_between_compartment(compt_1, compt_2)
                if btwn_compt is not None:
                    compt_1_i = self.matrix_name_to_index[compt_1.name]
                    compt_2_i = self.matrix_name_to_index[compt_2.name]
                    transition[compt_2_i, compt_1_i] \
                        = btwn_compt.parameter.get_initial_value()
        return transition

    def construct_matrix(self):
        """Construct the NGM with large domain for this model.

        Returns:
            np.array: the Next Generation Matrix.
        """
        T = self.construct_transmission_matrix()
        sigma = self.construct_transition_matrix()
        kl = np.dot(-T, np.linalg.inv(sigma))
        return kl

    def R0(self):
        """Calculate R0 from the NGM with large domain K_L for this
           model

        Returns:
            float: R0 for this model
        """
        ngm = self.construct_matrix()
        R0 = np.max(np.linalg.eig(ngm)[0])
        return R0.real