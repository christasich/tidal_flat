# tidal_flat

A zero-dimensional morphodynamic model of elevation change on a tidal platform.

## About this project

This project uses a zero-dimensional mass balance approach to model the elevation of a tidal platform over time. This work is based on previous studies (Krone, 1987; Allen, 1990; French, 1993; Temmerman et al., 2003, 2004) on the long-term evolution of marsh surfaces. We describe the fundamentals of the model briefly.

The depth of inundation of a tidal platform is defined as

```math
h(t) = \zeta(t) - \eta(t)
```

where $`\zeta(t)`$ is the water-surface elevation and $`\eta(t)`$ is the sediment-surface elevation. The rate of elevation change of the platform is then described as

```math
\frac{d\eta(t)}{dt} = \frac{dS_m(t)}{dt} + \frac{dS_o(t)}{dt} + \frac{dP(t)}{dt} + \frac{dM(t)}{dt}
```

The full description is described in Chris Tasich's dissertation and will be published soon.

## Source code

This project was built with `Python 3.10.9` using [Poetry](https://python-poetry.org/) to resolve dependencies. This project also requires [Git LFS](https://git-lfs.com/) which needs to be installed before cloning. You can install this with

```sh
git lfs install
```

Afterwards, use git to clone this repository into your computer.

```sh
git clone https://gitlab.jgilligan.org/chris/tidal_flat.git
```

Then navigate inside the cloned repository and install the environment.

```sh
cd tidal_flat/
poetry install
```

## Usage

*This section is also included as a notebook under the [example](example/) directory.*

Import pandas (to load the data) and the tidal_flat module.

```python
import pandas as pd
import tidal_flat as tf
```

Load a tidal time series into pandas. The time series must have a defined frequency or one that can be inferred from the data. We can use our sample data found in `example/tides.csv`.

*Note: This data set was based on five years of observations. A harmonic analysis was then used to create an extended tide record using [UTide](https://github.com/wesleybowman/UTide).*

```python
data = pd.read_csv("example/tides.csv", index_col="datetime", parse_dates=True, infer_datetime_format=True).squeeze()
data.index = pd.DatetimeIndex(data.index, freq="infer")
```

Because `pd.read_csv` does not set the frequency, we recreate the `pd.DatetimeIndex` and tell pandas to infer the frequency.

We can then create a tide object from this time series.

```python
tides = tf.Tides(data)
```

The tides class has some useful functions like `summarize` which calculates the tidal datums defined by [NOAA](https://tidesandcurrents.noaa.gov/datum_options.html). You can specify a frequency string to calculate at different intervals.

```python
tides.summarize(freq='A')
```
![](images/tides_summary.png)

There are also functions to change sea level, amplify the tides, or take slices of the data.

```python
tides.raise_sea_level(slr=0.005)        # 5mm/yr
tides.amplify(factor=1.25)              # factor of 1.25
tides.subset(start='2023', end='2025')  # subset three years of data from 2023 to 2025
```

Each function returns a copy of your tide object. We first raise sea level by $`3 mm \cdot yr^{-1}`$, then amplify the tides by an factor of $`1.25`$, and finally take a subset of the data from 2023 to 2025. This is useful for modeling changes to the tides or creating a subset without having to rebuild or reload them from scratch! These can also be chained together like this

```python
tides = tides.raise_sea_level(slr=0.003).amplify(factor=1.25).subset(start='2023', end='2025')
```

Now, we initialize our platform.

```python
platform = tf.platform.Platform(time_ref=tides.start, elevation_ref=2.0)
```

The platform class mostly keeps track of the history of the platform. We have to evolve it before it can really tell us anything interesting!

*Note: Both arguments are optional, but it's sometimes useful for consistency between runs or if you want to extend a simulation by using the same tide curve from the beginning.*

We can then initialize and run our model by passing the tides and platform to our model class along with a handful of parameters. A number of these parameters are optional. It's sometimes useful to exclude the other annual rates (organic matter, compaction, deep subsidence) from the model. For instance, you may want to simulate relative sea level rise. In this case, compaction and subsidence should be set to zero since they will be captured through an increase in the tidal water levels.

```python
model = tf.model.Model(
    tides=tides,
    platform=platform,
    ssc=0.2,                    # g/L
    grain_diameter=2.5e-5,      # m
    bulk_density=1e3,           # g/cm^3
    grain_density=2.65e3,       # g/cm^3    [Optional] default is the density of a quartz grain
    org_sed = 0.0,              # m/yr      [Optional]
    compaction = 7e-3,          # m/yr      [Optional]
    deep_sub = 3e-3             # m/yr      [Optional]
)
model.run()
```
![](images/simulation.gif)

Once the model is finished we can get the results like this.

```python
results = model.summarize()
```

The `results` returns two dataframes in `Bunch` object.

`results.platform` shows the total aggradation, total subsidence, total net surface elevation change, and surface elevation at each time step. For this simulation, we included subsidence in `slr` to capture a relative rate of sea level rise. Because of this, the results show no subsidence.

![](images/platform.png)

We can plot our results to show the change during our simulation. Let's plot the elevation of the platform and high water levels by month. This will allow us to see the seasonal signal in our data. We'll need two additional packages (`matplotlib` and `seaborn`) to make plotting easier.

```python
import matplotlib.pyplot as plt
import seaborn as sns

freq = 'M'

monthly_platform = results.platform.resample(freq).mean()
monthly_tides = model.tides.summarize(freq)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(data=monthly_platform.elevation, color='black', label='platform', ax=ax)
sns.lineplot(data=monthly_tides[['MHHW', 'MHW']], ax=ax)    # only plot mean high water and mean higher high water
ax.get_legend().set_title('')   # remove legend title
```

![](images/results.png)

`results.inundations` shows the characteristics of each inundation along with a few diagnostic variables.

![](images/inundations.png)

## Future plans

This package is actively being developed. Only the bare essentials have been documented here. We have a handful of other classes and functions that we have yet to expose. For instance, you may want to run a variety of platform conditions (I know we did!). Organizing this and keeping tracking of the results can become tedious. We use a combination of multiprocessing and yaml based configuration files to accomplish this. We've also implemented logging through [loguru](https://github.com/Delgan/loguru) that we have yet to document.

## License

[MIT](LICENSE)


## Report an issue / Ask a question
Use the [GitLab repository Issues](https://gitlab.jgilligan.org/chris/tidal_flat/-/issues).

## Sources

Allen, J. R. L. (Nov. 1, 1990). “Salt-Marsh Growth and Stratification: A Numerical Model with Special Reference to the Severn Estuary, Southwest Britain”. In: Marine Geology 95.2, pp. 77–96.

French, Jonathan R. (1993). “Numerical Simulation of Vertical Marsh Growth and Adjustment to Accelerated Sea-Level Rise, North Norfolk, U.K.” In: Earth Surface Processes and Landforms 18.1, pp. 63–81.

Krone, R.B. (1987). “A Method for Simulating Marsh Elevations”. In: Coastal Sediments. A Specialty Conference on Advances in Understanding of Coastal Sediment Processes. New Orleans, Louisiana: American Society of Civil Engineers, pp. 316–323

Temmerman, S. et al. (Jan. 15, 2003). “Modelling Long-Term Tidal Marsh Growth under Changing Tidal Conditions and Suspended Sediment Concentrations, Scheldt Estuary, Belgium”. In: Marine Geology 193.1, pp. 151–169.

Temmerman, S. et al. (Nov. 30, 2004). “Modelling Estuarine Variations in Tidal Marsh Sedimentation: Response to Changing Sea Level and Suspended Sediment Concentrations”. In: Marine Geology 212.1, pp. 1–19.
