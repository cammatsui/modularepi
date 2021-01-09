"""
Test sim results for transition, compartment, and model modules.
"""
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd() + '/../modular_epi')
os.chdir('../modular_epi')
from transition import *
from compartment import *
from model import *
from metapopulation_model import *
import unittest
import numpy as np

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Create compartment, OO SEIR model for testing
        self.test_meta = False

        # Simulation parameters for all models
        self.length = 100
        self.iters = 10
        self.pop = 1

        self.sigma_ = 1
        self.beta_ = 1
        self.gamma_ = 1/3

        # Divided parameters for manual model
        beta_ = self.beta_ / self.iters
        sigma_ = self.sigma_ / self.iters
        gamma_ = self.gamma_ / self.iters

        def make_param(name, val):
            return Parameter(name, np.zeros(self.length*self.iters) + val)
        self.sigma = make_param("sigma", self.sigma_)
        self.beta = make_param("beta", self.beta_)
        self.gamma = make_param("gamma", self.gamma_)

        # Set up compartments
        self.S = SusceptibleCompartment("Susceptible", self.pop,
            init_val = self.pop * 0.99, length = self.length,
            iters = self.iters)
        self.E = DiseaseCompartment("Exposed", init_val = self.pop * 0.01,
            length = self.length, iters = self.iters)
        self.I = DiseaseCompartment("Infectious", length = self.length,
            iters = self.iters)
        self.R = DiseaseCompartment("Recovered", length = self.length,
            iters = self.iters)
        self.S.add_transmission(self.I, self.pop, self.beta)
        self.S.add_transition_at_infection(self.E)
        self.E.add_transition(self.I, self.sigma)
        self.I.add_transition(self.R, self.gamma)
    
        # Set up OO model
        seir = CompartmentalModel(length = self.length, iters = self.iters)
        seir.add_susceptible_compartment("Susceptible", self.pop * 0.99,
            self.pop)
        seir.add_disease_compartment("Exposed", 
            init_val = self.pop * 0.01)
        seir.add_disease_compartment("Infectious")
        seir.add_disease_compartment("Recovered")
        # Add transitions
        seir.add_transmission("Susceptible", "Infectious", self.pop,
            ("beta", self.beta_))
        seir.add_transition_at_infection("Susceptible", "Exposed")
        seir.add_transition("Exposed", "Infectious", ("sigma", 
            self.sigma_))
        seir.add_transition("Infectious", "Recovered", ("gamma", 
            self.gamma_))

        seir.run()
        self.seir_outputs = seir.get_current_metrics()

        # Set up metapopulation model
        if self.test_meta:
            seir_m = MetapopulationModel(1, self.length, self.iters)
            seir_m.add_susceptible_compartment("Susceptible", [self.pop * 0.99],
                                            [self.pop])
            seir_m.add_disease_compartment("Exposed", [self.pop * 0.01])
            seir_m.add_disease_compartment("Infectious")
            seir_m.add_disease_compartment("Recovered")
            # Add transitions
            seir_m.add_transmission("Susceptible", "Infectious", [self.pop],
                                    ("beta", self.beta_), np.array([1, 1]))
            seir_m.add_transition_at_infection("Susceptible", "Exposed")
            seir_m.add_transition("Exposed", "Infectious", ("sigma", [self.sigma_]))
            seir_m.add_transition("Infectious", "Recovered", 
                ("gamma", [self.gamma_]))

            seir_m.run()
            self.seir_m_outputs = seir_m.get_current_metrics()

        # Manual model
        self.S_m = np.zeros(self.iters * self.length)
        self.E_m = np.zeros(self.iters * self.length)
        self.I_m = np.zeros(self.iters * self.length)
        self.R_m = np.zeros(self.iters * self.length)


        self.S_m[0] = self.pop * 0.99
        self.E_m[0] = self.pop * 0.01

        # Run models
        for t in range(1, self.iters * self.length):
            # Manual model
            foi = beta_ * self.S_m[t-1] * self.I_m[t-1] / self.pop
            rate_s2e = foi
            rate_e2i = sigma_ * self.E_m[t-1]
            rate_i2r = gamma_ * self.I_m[t-1]

            self.S_m[t] = self.S_m[t-1] - foi
            self.E_m[t] = self.E_m[t-1] + foi - rate_e2i
            self.I_m[t] = self.I_m[t-1] + rate_e2i - rate_i2r
            self.R_m[t] = self.R_m[t-1] + rate_i2r
            
            self.S.iterate_time()
            self.E.iterate_time()
            self.I.iterate_time()
            self.R.iterate_time() 

            # Compartment model
            self.S.transition_all()
            self.E.transition_all()
            self.I.transition_all()
            self.R.transition_all()

        self.S.values = self.S.values[range(0, self.length * self.iters,
            self.iters)]
        self.E.values = self.E.values[range(0, self.length * self.iters,
            self.iters)]
        self.I.values = self.I.values[range(0, self.length * self.iters,
            self.iters)]
        self.R.values = self.R.values[range(0, self.length * self.iters,
            self.iters)]

        self.S_m = self.S_m[range(0, self.length * self.iters, self.iters)]
        self.E_m = self.E_m[range(0, self.length * self.iters, self.iters)]
        self.I_m = self.I_m[range(0, self.length * self.iters, self.iters)]
        self.R_m = self.R_m[range(0, self.length * self.iters, self.iters)]
        
    def test_SEIR_compartments(self):
        """
        Test the OO SEIR model against a manually coded one. Checks that all
            timeseries are the same.
        """
        # Check equality for all time series
        self.assertTrue(np.alltrue(self.S_m == self.S.values))
        self.assertTrue(np.alltrue(self.E_m == self.E.values))
        self.assertTrue(np.alltrue(self.I_m == self.I.values))
        self.assertTrue(np.alltrue(self.R_m == self.R.values))
    
    def test_SEIR_model(self):
        """
        Test model class construction vs manual SEIR model.
        """
        self.assertTrue(np.alltrue(self.seir_outputs
            ['Susceptible'] == self.S_m))
        self.assertTrue(np.alltrue(self.seir_outputs['Exposed'] == self.E_m))
        self.assertTrue(np.alltrue(self.seir_outputs['Infectious'] == self.I_m))
        self.assertTrue(np.alltrue(self.seir_outputs['Recovered'] == self.R_m))
    
    def test_SEIR_meta_model(self):
        """
        Test metapopulation model with one population.
        """
        if self.test_meta:
            self.assertTrue(np.alltrue(self.seir_m_outputs
                ['Susceptible_0'] == self.S_m))
            self.assertTrue(np.alltrue(self.seir_m_outputs
                ['Exposed_0'] == self.E_m))
            self.assertTrue(np.alltrue(self.seir_m_outputs
                ['Infectious_0'] == self.I_m))
            self.assertTrue(np.alltrue(self.seir_m_outputs
                ['Recovered_0'] == self.R_m))

        
if __name__ == '__main__':
    unittest.main()