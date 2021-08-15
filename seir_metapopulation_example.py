from modularepi.metapopulation_model import MetapopulationModel
from modularepi.ngm import NGM
import matplotlib.pyplot as plt
import numpy as np

def main():
    # For all arrays, index 0 is children, index 1 is adults.
    NUM_POPS = 2
    POP_SIZES = [200, 1000]
    SIMULATION_DAYS = 365
    ITERATIONS_PER_DAY = 24 # Simulation iterations per day.
    INIT_INFECTIOUS = [5, 20]
    INIT_SUSCEPTIBLE = [POP_SIZES[0]-INIT_INFECTIOUS[0],
                        POP_SIZES[1]-INIT_INFECTIOUS[1]]

    # Model parameters.
    BETA_VAL = 0.3 # Baseline transmission rate is one parameter. Individual
    # group-to-group contact rates are controleld by the mixing matrix.
    SIGMA_VALS = [0.1, 0.3]
    GAMMA_VALS = [0.2, 0.1]
    # The mixing matrix gives the relative contact rates between each pair of
    # groups.
    MIXING_MATRIX = [
        [1.5, 1.2],
        [0.6, 1.0]]

    seir_metapopulation_model = MetapopulationModel(NUM_POPS, SIMULATION_DAYS,
                                                    ITERATIONS_PER_DAY)
                                
    # Add the compartments.
    seir_metapopulation_model.add_susceptible_compartment('S', INIT_SUSCEPTIBLE,
                                                          POP_SIZES)
    seir_metapopulation_model.add_disease_compartment('E')
    seir_metapopulation_model.add_disease_compartment('I',
                                                      init_vals=INIT_INFECTIOUS)
    seir_metapopulation_model.add_disease_compartment('R')

    # Add the transitions and transmissions.
    seir_metapopulation_model.add_transmission('S', 'I', POP_SIZES, 
                                               ('beta', BETA_VAL),
                                               MIXING_MATRIX)
    seir_metapopulation_model.add_transition_at_infection('S', 'E')
    seir_metapopulation_model.add_transition('E', 'I', ('sigma', SIGMA_VALS))
    seir_metapopulation_model.add_transition('I', 'R', ('gamma', GAMMA_VALS))

    # Run the model.
    seir_metapopulation_model.run()

    # Calculate and print the model's R0 with the Next Generation Matrix (NGM).
    seir_ngm = NGM(seir_metapopulation_model)
    print(seir_ngm.R0())

if __name__ == '__main__':
    main()