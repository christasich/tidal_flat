# tidal_flat

A zero-dimensional morphodynamic model of elevation change on a tidal platform.

## About this project

This project uses a zero-dimensional mass balance approach to model the elevation of a tidal platform over time. This work is based on previous studies (Krone, 1987; Allen, 1990; French, 1993; Temmerman et al., 2003, 2004) on the long-term evolution of marsh surfaces. We describe the fundamentals of the model briefly.

The depth of inundation of a tidal platform is defined as

```math
h(t) = \zeta(t) - \eta(t)
```

where $`\zeta(t)`$ is the water-surface elevation and $\eta(t)`$ is the sediment-surface elevation. The rate of elevation change of the platform is then described as

```math
\frac{d\eta(t)}{dt} = \frac{dS_m(t)}{dt} + \frac{dS_o(t)}{dt} + \frac{dP(t)}{dt} + \frac{dM(t)}{dt}
```

The full description is described in Chris Tasich's dissertation and will be published soon.

### Source code

Use git to clone this repository into your computer.

```
git clone https://gitlab.jgilligan.org/chris/tidal_flat.git
```

### Usage

Import the module and dependencies.

```python
import pandas as pd
import tidal_flat as tf
```

Load a tidal time series into pandas. The time series must have a defined frequency or one that can be inferred from the data. We can use our sample data found in `example/tides.csv`.

*Note: This data set was based on five years of observations. A harmonic analysis was then used to create an extended tide record using [UTide](https://github.com/wesleybowman/UTide).*

```python
data = pd.read_csv('example/tides.csv', index_col='datetime', parse_dates=True, infer_datetime_format=True).squeeze()
```

We can then create a tide object from this time series.

```python
tides = tf.Tides(data)
```

The tides class has some useful functions like `summarize` which calculates the tidal datums defined by [NOAA](https://tidesandcurrents.noaa.gov/datum_options.html). You can specify a frequency string to calculate at different intervals.

```python
annual = tides.summarize(freq='A')
```

There are also functions to change sea level, amplify the tides, or take slices of the data.

```python
tides = tides.raise_sea_level(slr=0.005)
tides = tides.amplify(af=1.5)
tides = tides.subset(start='2023', end='2025')
```

Each function returns a copy of your tide object. We first raise sea level by $`5 mm \cdot yr^{-1}`$, then amplify the tides by an annual factor of $`1.5`$, and finally take a subset of the data from 2023 to 2025. This is useful for modeling changes to the tides or creating a subset without having to rebuild or reload them from scratch! These can also be chained together like this

```python
tides = tides.raise_sea_level(slr=0.005).amplify(af=1.5).subset(start='2023', end='2025')
```

Finally, we initialize our platform.

```python
platform = tf.platform.Platform(time_ref=tides.start, elevation_ref=1.5)
```

The platform class mostly keeps track of the history of the platform. We have to evolve it before it can really tell us anything interesting!

*Note: Both arguments are optional, but it's sometimes useful for consistency between runs or if you want to extend a simulation by using the same tide curve from the beginning.*

We can then initialize our model by passing the tides and platform to our model class along with a handful of parameters.

```python
model = tf.model.Model(
    tides=tides,
    platform=platform,
    ssc=0.2,                    # g/L
    grain_diameter=2.5e-5,      # m
    bulk_density=1e3,           # g/cm^3
    grain_density=2.65e3        # g/cm^3    [Optional] default is the density of a quartz grain
    org_sed = 0.0,              # m/yr      [Optional]
    compaction = 0.0,           # m/yr      [Optional]
    deep_sub = 0.0              # m/yr      [Optional]
)
```
A number of the parameters are optional. It's sometimes useful to exclude the other annual rates (organic matter, compaction, deep subsidence) from the model. For instance, you may want to simulate relative sea level rise. In this case, compaction and subsidence should be set to zero since they will be captured through an increase in the tidal water levels.

Finally, we can run our model using,

```python
model.run()
```
![](images/simulation.gif)

and calculate the results with

```python
results = model.summarize()
```

The `results` returns two dataframes in `Bunch` object. `results.platform` shows the annual total aggradation, total subsidence, total net surface elevation change, and final surface elevation. `results.inundations` shows the characteristics of each inundation along with a few diagnostic variables.

## Future plans

This package is actively being developed. Only the bare essentials have been documented here. We have a handful of other classes and functions that we have yet to expose. For instance, you may want to run a variety of platform conditions (I know we did!). Organizing this and keeping tracking of the results can become tedious. We use a combination of multiprocessing and yaml based configuration files to accomplish this. We've also implemented logging through [loguru](https://github.com/Delgan/loguru) that we have yet to document

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
