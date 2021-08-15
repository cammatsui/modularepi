from modularepi.model import CompartmentalModel
from modularepi.ngm import NGM
import matplotlib.pyplot as plt
import numpy as np

def main():
    POP_SIZE = 1000
    SIMULATION_DAYS = 365
    ITERATIONS_PER_DAY = 24 # Simulation iterations per day.
    INIT_INFECTIOUS_POP = 1

    # Model parameters.
    BETA_VAL = 0.3
    SIGMA_VAL = 0.5
    GAMMA_VAL = 0.1

    seir_model = CompartmentalModel(SIMULATION_DAYS, ITERATIONS_PER_DAY)

    # Add the compartments.
    seir_model.add_susceptible_compartment('S', init_val=POP_SIZE-INIT_INFECTIOUS_POP, N=POP_SIZE)
    seir_model.add_disease_compartment('E')
    seir_model.add_disease_compartment('I', init_val=INIT_INFECTIOUS_POP)
    seir_model.add_disease_compartment('R')

    # Add the transitions and transmissions.
    seir_model.add_transmission('S', 'I', POP_SIZE, ('beta', BETA_VAL))
    seir_model.add_transition_at_infection('S', 'E')
    seir_model.add_transition('E', 'I', ('sigma', SIGMA_VAL))
    seir_model.add_transition('I', 'R', ('gamma', GAMMA_VAL))

    # Run the model.
    seir_model.run()

    # Calculate and print the model's R0 with the Next Generation Matrix (NGM).
    seir_ngm = NGM(seir_model)
    print(seir_ngm.R0())

    # Plot the simulation outputs.
    plt.plot(np.arange(SIMULATION_DAYS), seir_model.get_current_metrics()['I'])
    plt.plot(np.arange(SIMULATION_DAYS), seir_model.get_cumulative()['I'])
    plt.show()

if __name__ == '__main__':
    main()