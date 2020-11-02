import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd() + '/../modular_epi')
os.chdir('../modular_epi')
from transition import *
from compartment import *
from model import *
from ngm import *
import unittest
import numpy as np


class TestNGM(unittest.TestCase):

    def setUp(self):
        # Set up multiple models for testing
        # Parameters:
        self.length = 100
        self.iters = 10
        self.pop = 100
        self.beta = 1.5
        self.gamma = 1

        # SIR model
        self.sir = CompartmentalModel(length = self.length, iters = self.iters)
        self.sir.add_susceptible_compartment("Susceptible", self.pop * 0.99,
            self.pop)
        self.sir.add_disease_compartment("Infectious",
            init_val = self.pop * 0.01)
        self.sir.add_disease_compartment("Recovered")
        self.sir.add_transmission("Susceptible", "Infectious", 
            self.pop, ("beta", self.beta))
        self.sir.add_transition_at_infection("Susceptible", "Infectious")
        self.sir.add_transition("Infectious", "Recovered",
            ("gamma", self.gamma))

        self.ngm_sir = NGM(self.sir)
        
        # SEIR with two exposed compartments from NGM paper
        self.p = 0.3 # proportion going to E_1
        self.nu_1 = 1 / 3
        self.nu_2 = 1 / 5
        self.mu = 1 / 100

        self.seir_paper = CompartmentalModel(length = self.length,
            iters = self.iters)
        self.seir_paper.add_susceptible_compartment("Susceptible",
            self.pop * 0.99, self.pop, self_param = ("mu", self.mu))
        self.seir_paper.add_disease_compartment("Exposed_1",
            init_val = self.pop * 0.01, self_param = ("mu", self.mu))
        self.seir_paper.add_disease_compartment("Exposed_2",
            self_param = ("mu", self.mu))
        self.seir_paper.add_disease_compartment("Infectious",
            self_param = ("mu", self.mu))
        self.seir_paper.add_disease_compartment("Recovered",
            self_param = ("mu", self.mu))

        self.seir_paper.add_transmission("Susceptible", "Infectious",
            self.pop, ("beta", self.beta))
        self.seir_paper.add_transition_at_infection("Susceptible", "Exposed_1",
            proportion_param = ("p", self.p))
        self.seir_paper.add_transition_at_infection("Susceptible", "Exposed_2",
            proportion_param = ("(1-p)", (1-self.p)))
        self.seir_paper.add_transition("Exposed_1", "Infectious",
            ("nu_1", self.nu_1))
        self.seir_paper.add_transition("Exposed_2", "Infectious",
            ("nu_2", self.nu_2))
        self.seir_paper.add_transition("Infectious", "Recovered",
            ("gamma", self.gamma))
        
        self.ngm_seir_paper = NGM(self.seir_paper)

        ## SEIR sexually transmitted, from paper
        pop_1 = 100
        pop_2 = 160
        beta_1_ = 1.5
        beta_2_ = 1.2
        beta_1_param = ("beta_1", beta_1_)
        beta_2_param = ("beta_2", beta_2_)
        nu_1_ = 0.8
        nu_2_ = 1.3
        nu_1_param = ("nu_1", nu_1_)
        nu_2_param = ("nu_2", nu_2_)
        gamma_1_ = 1.1
        gamma_2_ = 0.7
        gamma_1_param = ("gamma_1", gamma_1_)
        gamma_2_param = ("gamma_2", gamma_2_)
        mu_param = ("mu", self.mu)
        sti_seir = CompartmentalModel(length = self.length, iters =
            self.iters)
        sti_seir.add_susceptible_compartment("Susceptible_1", pop_1 * 0.99,
            pop_1, self_param = mu_param)
        sti_seir.add_susceptible_compartment("Susceptible_2", pop_2 * 0.99,
            pop_2, self_param = mu_param)
        sti_seir.add_disease_compartment("Exposed_1", init_val = pop_1 * 0.01,
            self_param = mu_param)
        sti_seir.add_disease_compartment("Exposed_2", init_val = pop_2 * 0.01,
            self_param = mu_param)
        sti_seir.add_disease_compartment("Infectious_1", self_param = mu_param)
        sti_seir.add_disease_compartment("Infectious_2", self_param = mu_param)
        sti_seir.add_disease_compartment("Recovered_1", self_param = mu_param)
        sti_seir.add_disease_compartment("Recovered_2", self_param = mu_param)

        # add transmissions and transitions
        sti_seir.add_transmission("Susceptible_1", "Infectious_2", pop_2,
            beta_1_param)
        sti_seir.add_transmission("Susceptible_2", "Infectious_1", pop_1,
            beta_2_param)
        sti_seir.add_transition_at_infection("Susceptible_1", "Exposed_1")
        sti_seir.add_transition_at_infection("Susceptible_2", "Exposed_2")

        sti_seir.add_transition("Exposed_1", "Infectious_1", nu_1_param)
        sti_seir.add_transition("Exposed_2", "Infectious_2", nu_2_param)
        sti_seir.add_transition("Infectious_1", "Recovered_1", gamma_1_param)
        sti_seir.add_transition("Infectious_2", "Recovered_2", gamma_2_param)

        self.ngm_sti_seir = NGM(sti_seir)
        self.sti_seir_R0 = np.sqrt((nu_1_*nu_2_*beta_1_*beta_2_) \
            / ((nu_1_+self.mu) * (nu_2_+self.mu) * (gamma_1_+self.mu) * \
            (gamma_2_+self.mu)))

        
        # Test hurricane NGM: https://www.medrxiv.org/content/10.1101/2020.08.
        # 07.20170555v1.full.pdf

        # gamma = 1/D, sigma 1/Z
        hurr_seir = CompartmentalModel(length = self.length, iters = self.iters)
        hurr_seir.add_susceptible_compartment("Susceptible", self.pop * 0.99,
            self.pop)
        hurr_seir.add_disease_compartment("Exposed", init_val = self.pop * 0.01)
        hurr_seir.add_disease_compartment("Infectious_R")
        hurr_seir.add_disease_compartment("Infectious_U")
        hurr_seir.add_disease_compartment("Recovered")
        
        # transmissions and transitions
        hurr_gamma = 0.3
        hurr_beta = 1.9
        hurr_mu = 0.69
        hurr_alpha = 0.44827
        hurr_sigma = 0.3

        hurr_seir.add_transmission("Susceptible", "Infectious_R", self.pop,
            ("beta", hurr_beta))
        hurr_seir.add_transmission("Susceptible", "Infectious_U", self.pop,
            ("beta*mu", hurr_beta*hurr_mu))
        hurr_seir.add_transition_at_infection("Susceptible", "Exposed")
        hurr_seir.add_transition("Exposed", "Infectious_R",
            ("alpha*sigma", hurr_alpha*hurr_sigma))
        hurr_seir.add_transition("Exposed", "Infectious_U",
            ("(1-alpha)*sigma", (1-hurr_alpha)*hurr_sigma))
        hurr_seir.add_transition("Infectious_R", "Recovered",
            ("gamma", hurr_gamma))
        hurr_seir.add_transition("Infectious_U", "Recovered",
            ("gamma", hurr_gamma)
        )
    
        hurr_seir.run()
        self.ngm_hurr_seir = NGM(hurr_seir)
        self.hurr_seir_R0 = hurr_beta * (1/hurr_gamma) * (hurr_alpha + (hurr_mu\
            * (1-hurr_alpha))) 
        
    def test_SIR_ngm_compartments(self):
        compt_names = {c.name for c in self.ngm_sir._get_ngm_compartments()}
        self.assertEqual(compt_names, {"Infectious"})
    
    def test_SEIR_paper_ngm_compartments(self):
        compt_names = {c.name for c in self.ngm_seir_paper.\
            _get_ngm_compartments()}
        self.assertEqual(compt_names, {"Exposed_1", "Exposed_2", "Infectious"})
        
    def test_SEIR_paper_ngm_transition(self):
        transition = self.ngm_seir_paper.construct_transition_matrix()
        paper_transition = np.array([[-(self.nu_1 + self.mu), 0, 0],
                                     [0, -(self.nu_2 + self.mu), 0],
                                     [self.nu_1, self.nu_2, -(self.gamma + \
                                         self.mu)]])
        self.assertTrue(np.alltrue(transition == paper_transition))
    
    def test_SEIR_paper_ngm_transmission(self):
        transmission = self.ngm_seir_paper.construct_transmission_matrix()
        paper_transmission = np.array([[0, 0, self.p * self.beta],
                                      [0, 0, (1-self.p) * self.beta],
                                      [0, 0, 0]])
        self.assertTrue(np.alltrue(transmission == paper_transmission))
    
    def test_SEIR_paper_R0(self):
        R0 = self.ngm_seir_paper.R0()
        R0_paper = (((self.p * self.nu_1) / (self.nu_1 + self.mu)) + \
            (((1-self.p) * self.nu_2) / (self.nu_2 + self.mu))) * self.beta \
            / (self.gamma + self.mu)
        self.assertAlmostEqual(R0, R0_paper)
    
    def test_SIR_R0(self):
        R0 = self.ngm_sir.R0()
        true_R0 = self.beta / self.gamma
        self.assertAlmostEqual(R0, true_R0)
    
    def test_sti_seir_R0(self):
        R0 = self.ngm_sti_seir.R0()
        true_R0 = self.sti_seir_R0
        self.assertAlmostEqual(R0, true_R0)
    
    def test_hurr_seir_R0(self):
        R0 = self.ngm_hurr_seir.R0()
        true_R0 = self.hurr_seir_R0
        self.assertAlmostEqual(R0, true_R0)
        print(R0, true_R0)
        
if __name__ == '__main__':
    unittest.main()