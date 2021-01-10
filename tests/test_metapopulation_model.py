import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd() + '/../modularepi')
os.chdir('../modularepi')
from transition import *
from compartment import *
from model import *
from metapopulation_model import *
import unittest
import numpy as np

class TestMetapopulationModel(unittest.TestCase):

    def setUp(self):
        length = 200
        iters = 10
        c = 3
        p1 = 50
        p2 = 100
        p3 = 25
        self.mat = np.array([
            [0.5, 1.0, 2.5],
            [3.2, 1.3, 0.7],
            [0.2, 2.3, 3.8]
        ])
        self.beta_val = 1.5
        self.sigma_val = 0.6
        self.gamma_val = 1

        self.model = MetapopulationModel(c, length, iters)
        self.model.add_susceptible_compartment(
            "Susceptible", 
            [p1 * 0.99, p2 * 0.99, p3 * 0.99],
            [p1, p2, p3]
        )
        self.model.add_disease_compartment(
            "Exposed",
            [p1 * 0.01, p2 * 0.01, p3 * 0.01]
        )
        self.model.add_disease_compartment("Infectious")
        self.model.add_disease_compartment("Recovered")

        self.model.add_transmission(
            "Susceptible", "Infectious", [p1, p2, p3],
            ("Beta", self.beta_val * np.ones(3)), self.mat
        )
        self.model.add_transition_at_infection("Susceptible", "Exposed")
        self.model.add_transition("Exposed", "Infectious",
                                  ("Sigma", self.sigma_val * np.ones(3)))
        self.model.add_transition("Infectious", "Recovered",
                                  ("Gamma", self.gamma_val * np.ones(3)))
    
    def test_parse_params_none(self):
        parsed = self.model._parse_params(None, 0.5)
        self.assertEqual(parsed, [("", 0.5), ("", 0.5), ("", 0.5)])
    
    def test_parse_params_wronglength(self):
        with self.assertRaises(ValueError):
            self.model._parse_params(("Test", [0, 1]), 0)
        
    def test_parse_params_formatted(self):
        params = ("Beta", [1, 1.5, 2])
        params_parsed = [("Beta_0", 1), ("Beta_1", 1.5), ("Beta_2", 2)]
        self.assertEqual(params_parsed, self.model._parse_params(params, 0))

    def test_compartments(self):
        compts = set()
        for i in range(3):
            compts.add(f"Susceptible_{i}")
            compts.add(f"Exposed_{i}")
            compts.add(f"Infectious_{i}")
            compts.add(f"Recovered_{i}")
        self.assertEqual(self.model.compartments.keys(), compts)
    
    def test_transitions(self):
        transitions = []
        model_transition_names = [str(transition) for transition in 
            self.model.transitions]
        for i in range(3):
            transitions.append(f"Exposed_{i} -> Infectious_{i}")
            transitions.append(f"Infectious_{i} -> Recovered_{i}")
        self.assertEqual(transitions.sort(), model_transition_names.sort())

            
        

if __name__ == '__main__':
    unittest.main()
