
# ModularEpi

## **Table of Contents**

 - [**Overview**](#overview)
 - [**Basic Structure**](#basic-structure)
 - [**Calculating _R0_ with the Next Generation Matrix**](#calculating-r0-with-the-next-generation-matrix)
 - [**Single-Population SEIR Model Example**](#single-population-seir-model-example)
 - [**Getting Simulation Output**](#getting-simulation-output)
 - [**Metapopulation SEIR Model Example**](#metapopulation-seir-model-example)

 ## **Overview**

ModularEpi is a Python framework to create compartmental epidemiological models and run simulations on them. The framework includes support for both single-population and metapopulation models, and implements the [Next Generation Matrix](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2871801/) method to compute *R0* for models.

## **Basic Structure**

The framework uses 5 basic object types to create `CompartmentalModel` objects:
1. `DiseaseCompartment`

    A `DiseaseCompartment` is compartment in a model that contains individuals. For example, our model may include _Infectious_, _Exposed_,  and _Removed_ `DiseaseCompartment`s. 

    To add a `DiseaseCompartment` to the `CompartmentalModel` `model`, we call
    ```py
    model.add_disease_compartment(name, initial_value, self_param=self_parameter)
    ```
    where `name` is a string that is the (unique) name of the compartment, `intial_value` is the initial occupancy of the compartment, and `self_parameter` is a tuple of the form 
    ```python
    self_parameter = (self_param_name, self_param_value)
    ```
    where the occupancy of the compartment changes by the self parameter's value on each day, for example, as a birth or death rate. By default, a `DiseaseCompartment`'s self parameter has no name and has a value of 0.

1. `SusceptibleCompartment`

    A `SusceptibleCompartment` is a `DiseaseCompartment` that contains susceptible individuals.
    
    To add a `SusceptibleCompartment` to `model`, we call
    ```python
    model.add_susceptible_comaprtment(name, initial_value, N, self_param=self_parameter)
    ```
    where `N` is the total size of this `SusceptibleCompartment`'s associated population.


1. `Transition`

    A `Transition` connects two compartments and has an associated `Parameter`. It indicates that individuals should transition from one of these compartments to the other based on the given parameter value.

    To add a transition to our model between, for example, the `DiseaseCompartments` with names `'E'` and `'I'`, we call:
    ```python
    model.add_transition('E', 'I', rate_parameter)
    ```
    where `rate_parameter` is a tuple of the form 
    ```python
    rate_parameter = (parameter_name, parameter_value)
    ```
    indicating the rate of transition between the compartments.

    We can additionally add multiple transitions out of one compartment, or multiple into another compartment.

1. `Transmission`

    A `Transmission` connects a `SusceptibleCompartment` and a `DiseaseCompartment` with an associated `Parameter` and indicates that individuals in the given `SusceptibleCompartment` are infected by individuals in the given `DiseaseCompartment` at the rate given by the parameter value. 
    
    To add a `Transmission` to `model`, say where the `SusceptibleCompartment` with name `'S'` is infected by the `DiseaseCompartment` with name `'I'` and then transitions to a compartment with name `'E'`, we have to add both a `Transmission` and a special type of `Transition` called a `transition_at_infection`:
    ```python
    model.add_transmission('S', 'I', N, transmission_parameter)
    model.add_transition_at_infection('S', 'E', proportion_parameter)
    ```
    The `transmission_parameter` is a tuple (in the form of the other parameter tuples) where the rate value indicates the baseline transmission rate of `'I'` onto `'S'`. The `proportion_parameter`'s value indicates the proportion of infected individuals at each timesetp that should transition to `'E'`. By default, this proportion is 1, but if we would like to implement multiple types of exposed compartments such that infected individuals transition into each of them with chosen probabilities, we can add multiple exposed compartments and multiple transitions at infection.

## **Calculating _R0_ with the Next Generation Matrix**

To calculate _R0_ for a `CompartmentalModel` `model`, we create a new `NGM` object from the model:

```python
from modularepi.ngm import NGM

model_ngm = NGM(model)
R0 = model_ngm.R0()
```

## **Single-Population SEIR Model Example**

Below is an example implementation for a simple closed-system SEIR model with no births or deaths. In this case, `beta_val` is the transmission rate, `sigma_val` the rate of transition from the exposed compartment `'E'` to the infectious compartment `'I'`, and `gamma_val` the recovery/removal rate from `'I'` to the removed compartment `'R'`. 

```python
from modularepi.model import CompartmentalModel
from modularepi.ngm import NGM

# Simulation parameters.
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
```

The code for this example is provided in ```seir_example.py```.

## **Getting Simulation Output**

We can receive three types of simulation results from a model:

1. Current compartment values on each day via ```model.get_current_metrics()```
1. Incident compartment values on each day (i.e., the change in a compartment's value on each day) via ```model.get_incidence()```
1. Cumulative compartment values on each day via ```model.get_cumulative()```

All three methods return a dictionary with keys as compartment names and values as arrays with the type of metric's value at each day (given by the array index).

##  **Metapopulation SEIR Model Example**

In a metapopulation model, we consider _c_ different metapopulations that follow the same disease progression but with different parameters. These metapopulations can infect each other (and themselves) at rates specified in the model. The below example is an SEIR model with two metapopulations (children and adults); each metapopulation infects itself and the other as specified by the mixing matrix, which gives the relative contact rates among and between groups, and each metapopulation progresses through the disease at different rates.

The example code can be found in `seir_metapopulation_example.py`.

```python
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
```