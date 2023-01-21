# tidal_platform

A zero-dimensional morphodynamic model of elevation change on a tidal platform.

## About this project

This project uses a zero-dimensional mass balance approach to model the elevation of a tidal platform over time. This work is based on previous studies (Krone, 1987; Allen, 1990; French, 1993; Temmerman et al., 2003, 2004) on the longterm evolution of marsh surfaces. We describe the fundamentals of the model briefly.

The depth of inundation of a tidal platform is defined as

```math
h(t) = \zeta(t) - \eta(t)
```

where $`\zeta(t)`$ is the water-surface elevation and $\eta(t)`$ is the sediment-surface elevation. The rate of elevation change of the platform is then described as

```math
\frac{d\eta(t)}{dt} = \frac{dS_m(t)}{dt} + \frac{dS_o(t)}{dt} + \frac{dP(t)}{dt} + \frac{dM(t)}{dt}
```

The full description can be found here.

### Source code

Use git to clone this repository into your computer.

```
git clone https://gitlab.jgilligan.org/chris/tidal_platform.git
```

### Usage

Import the module and dependencies.

```python
    import pandas as pd
    import tidal_flat as tf
```

Load a tidal time series into pandas. The time series must have a defined frequency or one that can be infered from the data.

```python
    tides = tf.Tides(data)
```

The tides class has some useful functions like summarize which calculates the tidal datums defined by [NOAA](https://tidesandcurrents.noaa.gov/datum_options.html). You can specify a frequency string to calculate at different intervals.

```python
    annual = tides.summarize(freq='A')
```

There are also functions to amplify the tides and change sea level.

```python
    tides = tides.raise_sea_level(slr=0.005)
    tides = tides.amplify(af=1.5)
    tides = tides.subset(start='2020', end='2030')
```

Each functions returns a copy of your tide object. We first raise sea level by $`5 mm \cdot yr^{-1}`$, then amplify the tides by a factor of $`1.5`$, and finally take a subset of the data from 2020 to 2030. This is useful for modeling changes to the tides or sample subsets without having to rebuild or reload them from scratch!

Finally, we initialize our platform.

```python
    platform = tf.platform.Platform(time_ref=tides.start, elevation_ref=0.0)
```

The platform class mostly keeps track of the history of the platform. We have to evolve it before it can really tell us anything interesting! We can then run our model by passing the tides and platform to our model class along with a handful of parameters.

```python

```


## License

[MIT](LICENSE)


# Report an issue / Ask a question
Use the [GitLab repository Issues](https://gitlab.jgilligan.org/chris/tidal_flat/-/issues).

## Sources

Allen, J. R. L. (Nov. 1, 1990). “Salt-Marsh Growth and Stratification: A Numerical Model with Special Reference to the Severn Estuary, Southwest Britain”. In: Marine Geology 95.2, pp. 77–96.

French, Jonathan R. (1993). “Numerical Simulation of Vertical Marsh Growth and Adjustment to Accelerated Sea-Level Rise, North Norfolk, U.K.” In: Earth Surface Processes and Landforms 18.1, pp. 63–81.

Krone, R.B. (1987). “A Method for Simulating Marsh Elevations”. In: Coastal Sediments. A Specialty Conference on Advances in Understanding of Coastal Sediment Processes. New Orleans, Louisiana: American Society of Civil Engineers, pp. 316–323

Temmerman, S. et al. (Jan. 15, 2003). “Modelling Long-Term Tidal Marsh Growth under Changing Tidal Conditions and Suspended Sediment Concentrations, Scheldt Estuary, Belgium”. In: Marine Geology 193.1, pp. 151–169.

Temmerman, S. et al. (Nov. 30, 2004). “Modelling Estuarine Variations in Tidal Marsh Sedimentation: Response to Changing Sea Level and Suspended Sediment Concentrations”. In: Marine Geology 212.1, pp. 1–19.
