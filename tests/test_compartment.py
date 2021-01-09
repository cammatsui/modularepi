import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd() + '/../modular_epi')
os.chdir('../modular_epi')
from transition import *
from compartment import *
from model import *
import unittest
import numpy as np

class TestCompartment(unittest.TestCase):

    def setUp(self):
        # Make simple SIR model
        self.iters = 10
        self.length = 100
        self.pop = 8000
        self.beta = 1.5
        self.gamma = 1
        self.S = SusceptibleCompartment(
            "Susceptible", self.pop, init_val=self.pop*0.99,
            length = self.length, iters = self.iters
        )
        self.I = DiseaseCompartment(
            "Infectious", init_val=self.pop*0.01, length=self.length,
            iters=self.iters
        )
        self.R = DiseaseCompartment(
            "Recovered", length=self.length, iters=self.iters
        )

        def make_param(name, val):
            return Parameter(name, np.zeros(self.iters*self.length) + val)

        beta_param = make_param('beta', self.beta)
        gamma_param = make_param('gamma', self.gamma)
        self.S.add_transmission(
            self.I, self.pop, beta_param
        )
        self.S.add_transition_at_infection(self.I)
        self.I.add_transition(
            self.R, gamma_param
        )
    
    def test_add_transition(self):
        # Test the transition from I to R
        self.assertEqual(self.I.transitions[0].parameter.value[0], self.gamma)
        self.assertEqual(self.I.transitions[0].parameter.name, "gamma")
        self.assertEqual(self.I.transitions[0].compartment, self.R)
        self.assertEqual(self.I.transitions[0].simulation_length,
                         self.length)
        self.assertEqual(self.I.transitions[0].intervals_per_day,
                         self.iters)
    
    def test_calculate_transition(self):
        # Test I to R transition
        self.I.iterate_time()
        self.assertEqual(
            self.I.calculate_transition(self.I.transitions[0]),
            (self.pop*0.01)*(self.gamma/self.iters)
        )
        self.I.reset()
    
    def test_transition_all(self):
        self.I.iterate_time()
        cur = self.I.current_value
        transition_val = self.I.calculate_transition(self.I.transitions[0])
        self.I.transition_all()
        self.I.iterate_time()
        self.assertEqual(
            transition_val, cur - self.I.current_value
        )
        self.S.reset()
        self.I.reset()
        self.R.reset()

    
    def test_add_value(self):
        self.S.iterate_time()
        cur_val = self.S.current_value
        self.S.iterate_time()
        self.S.add_value(100)
        self.S.iterate_time()
        self.assertEqual((cur_val + 100), self.S.current_value)
        self.S.reset()
    
    def test_add_transmission(self):
        # Test that S is infected by I
        transmission = self.S.transmissions[0]
        self.assertEqual(
            transmission.name, "Susceptible *> Infectious"
        )
        self.assertEqual(transmission.parameter.value[0], self.beta)
    
    # TODO: Fill out these tests!
    
    def test_change_transmission_parameter(self):
        pass

    def test_add_transition_at_infection(self):
        # Test that I is transition at infection for S
        transition_at_infection = self.S.transitions_at_infection[0]
        self.assertEqual(transition_at_infection.parameter.value[0], 1)
        self.assertEqual(
            transition_at_infection.name.split()[-1], "Infectious"
        )
    
    def test_change_transition_at_infection_proportion(self):
        pass

    def test_calculate_infected(self):
        pass
    
    def test_transition_infections(self):
        pass

    def test_reset(self):
        pass

if __name__ == '__main__':
    unittest.main()