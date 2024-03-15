# Holistic Wind Farm Control
Investigation of the impact of holistic control of turbines in a wind farm

## Files
- [exploration.ipynb](exploration.ipynb) - Exploration of the impact of parameters such as yaw, wind speed and wind direction on cost and utilisation
- [optimisation.ipynb](optimisation.ipynb) - Optimisation of yaw values for different wind and site conditions to reduce cost

## To investigate
- Increase speed and direction resolution
    - Fix oscillations around rated wind speed
- Larger wind farms
- Impact of varying baseline turbulence
- Real time steering values
    - CNN
    - [Lookup table/surrogate model](https://adaptive.readthedocs.io/en/latest/algorithms_and_examples.html#examples)
- Off design performance
    - Local variations in speed and direction
    - Measured using LiDAR
- Curtailment of front turbines as well as steering

## To do