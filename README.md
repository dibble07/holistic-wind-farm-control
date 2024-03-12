# Holistic Wind Farm Control
Investigation of the impact of holistic control of turbines in a wind farm

## Files
- [exploration.ipynb](exploration.ipynb) - Exploration of the impact of parameters such as yaw, wind speed and wind direction on cost and utilisation
- [optimisation.ipynb](optimisation.ipynb) - Optimisation of yaw values for different wind and site conditions to reduce cost

## To investigate
- Larger wind farms
- Real time steering values
    - CNN
    - [Lookup table/surrogate model](https://adaptive.readthedocs.io/en/latest/algorithms_and_examples.html#examples)
    - Low fidelity model
    - LiDAR adjustments
- Analyse time series wind resource
    - Impact of varying baseline turbulence
- Curtailment of front turbines as well as steering

## To do
- Drop plots about capacity factor
- Increase number of turbines
- Create own version of V80 turbine with high resolution with continuous gradients but no output below cut in speed