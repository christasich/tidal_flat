import contextlib
from dataclasses import dataclass, field, InitVar
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.utils import Bunch
from tqdm.auto import tqdm
import numpy as np

from scipy.interpolate import PchipInterpolator

# from .inundation import Inundation
from .utils import normalize_timeseries

from .tides import Tides
from .inundation import Inundation
from .platform import Platform


def equilibrate_platform(
    init_elevation: float,
    water_levels: pd.Series,
    mtr: float,
    rslr_rate: float,
    ssc: float,
    grain_diameter: float,
    bulk_density: float,
):
    tides = Tides(water_levels=water_levels)
    tides = tides.amplify(factor=mtr / tides.summary.MN)
    # models = []
    results = []
    year = 0
    prev_elevation = elevation = init_elevation

    while True:
        tides = tides.raise_sea_level(rslr_rate)
        platform = Platform(
            init_elevation=elevation,
            ssc=ssc,
            grain_diameter=grain_diameter,
            bulk_density=bulk_density,
        )
        tqdm_kwargs = {
            "desc": f"TR={mtr:.1f} m | SSC={ssc:.3f} g/L | RSLR={rslr_rate * 1000:.1f} mm | YEAR={year}",
            "leave": False,
            "position": 0,
        }
        model = Model(water_levels=tides.water_levels, platform=platform, tqdm_kwargs=tqdm_kwargs)
        model.run()
        # models.append(model)
        result = pd.concat(
            [
                model.results.totals.iloc[-1].rename(year),
                model.results.inundations.squeeze().rename(year),
                model.results.tides.squeeze().rename(year),
            ],
            keys=["totals", "inundations", "tides"],
        )
        results.append(result)
        elevation = model.results.totals.elevation.iloc[-1]
        delta = elevation - prev_elevation
        if delta < rslr_rate * 1.5:
            return pd.concat(results, axis=1).T
        prev_elevation = elevation
        year += 1


@dataclass
class Model:
    # input these fields
    tides: Tides
    platform: Platform
    _inundations: list = field(default_factory=list)
    solve_ivp_kwargs: dict = field(default_factory=dict)
    tqdm_kwargs: dict = field(default_factory=dict)

    # auto initialize these fields
    results: pd.DateOffset = field(init=False)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    period: pd.Timedelta = field(init=False)
    now: pd.Timestamp = field(init=False)
    n: int = 0
    freq: pd.offsets.BaseOffset = field(init=False)
    tz: Any = field(init=False)

    @property
    def remaining_cycles(self):
        return self.tides.cycles.loc[self.n:]

    @property
    def inundations(self):
        return pd.concat(self._inundations, axis=1).T

    def __post_init__(self) -> None:
        self.start = self.now = self.tides.start
        self.end = self.tides.end
        self.period = self.tides.period
        self.freq = self.tides.water_levels.index.freq
        self.tz = self.tides.water_levels.index.tz
        self.logger = logger.patch(lambda record: record["extra"].update(model_time=self.now))
        self.setup_pbar()

    # def setup_pbar(self, end_date=None, period=None):
    #     if end_date:
    #         self.end = pd.to_datetime(end_date)
    #     if period:
    #         self.end = self.start + pd.to_timedelta(period)
    #     if self.period < pd.Timedelta("365.2425D"):
    #         time_unit = pd.Timedelta("1D")
    #         unit = "Day"
    #     else:
    #         time_unit = pd.Timedelta("365.2425D")
    #         unit = "Year"
    #     tqdm_kwargs = {
    #         "total": round(self.period / time_unit, 2),
    #         "unit": unit,
    #         "dynamic_ncols": True,
    #         "maxinterval": 1,
    #         "smoothing": 0,
    #         "postfix": {"Elevation": f"{self.platform.init_elevation:.3f} ({0.0:+.3f}) m"},
    #     }
    #     if self.tqdm_kwargs:
    #         self.tqdm_kwargs |= tqdm_kwargs
    #     self.pbar = tqdm(**self.tqdm_kwargs)
    #     self.pbar.time_unit = time_unit

    def setup_pbar(self):
        tqdm_kwargs = {
            "total": len(self.remaining_cycles),
            "unit": 'cycles',
            "dynamic_ncols": True,
            "maxinterval": 1,
            "smoothing": 0,
            "postfix": {"Elevation": f"{self.platform.init_elevation:.3f} ({0.0:+.3f}) m"},
        }
        if self.tqdm_kwargs:
            self.tqdm_kwargs |= tqdm_kwargs
        self.pbar = tqdm(**self.tqdm_kwargs)

    def update_pbar(self, refresh=False):
        self.pbar.set_postfix(
            {"Elevation": f"{self.platform.state['elevation']:.3f} ({self.platform.state['net']:+.3f}) m"},
            refresh=False,
        )
        self.pbar.update(n=self.n - self.pbar.n)
        if refresh:
            self.pbar.refresh()

    # def update_pbar(self, refresh=False):
    #     new_n = round((self.now - self.start) / self.pbar.time_unit, 2)
    #     self.pbar.set_postfix(
    #         {"Elevation": f"{self.platform.state['elevation']:.3f} ({self.platform.state['net']:+.3f}) m"},
    #         refresh=False,
    #     )
    #     self.pbar.update(n=new_n - self.pbar.n)
    #     if refresh:
    #         self.pbar.refresh()

    def next_cycle(self):
        check = self.remaining_cycles.high > self.platform.surface
        return self.tides.cycles.loc[check.idxmax()] if check.any() else None

    def make_inundation(self, cycle: pd.Series) -> Inundation:
        subset = self.tides.water_levels.loc[cycle.start: cycle.end]
        subset = subset - self.platform.surface
        depth = PchipInterpolator(
            x=subset.index.astype(int) / 10**9,
            y=subset.values,
            extrapolate=False,
        )
        roots = depth.roots()
        if roots.size == 2:
            r1 = roots[0] + 1
            r2 = roots[1] - 1
        elif roots.size == 1:
            if roots[0] < cycle.slack.timestamp():
                r1 = roots[0] + 1
                r2 = cycle.end.timestamp()
            else:
                r1 = cycle.start.timestamp()
                r2 = roots[0] - 1
        elif roots.size == 0 and (subset > self.platform.surface).all():
            r1 = cycle.start.timestamp()
            r2 = cycle.end.timestamp()
        start = pd.to_datetime(r1, unit="s", origin="unix", utc=True).tz_convert(self.tz)
        end = pd.to_datetime(r2, unit="s", origin="unix", utc=True).tz_convert(self.tz)

        return Inundation(depth=depth, start=start, end=end)

    def step(self):
        cycle = self.next_cycle()
        if cycle is not None:
            inundation = self.make_inundation(cycle)
            self.process_inundation(inundation)
            self.now = inundation.end
            self.n = cycle.name + 1
            # self.process_cycle(cycle)
            self.update_pbar()
        else:
            self.platform.update(elapsed=self.end - self.now)
            self.now = self.end
            self.n = self.tides.cycles.index[-1] + 1
            self.update_pbar(refresh=True)

    # def process_cycle(self, cycle):
    #     inundation = self.make_inundation(cycle)
    #     self.process_inundation(inundation)

    #     self.now = inundation.end
    #     self.n = cycle.name + 1

    def process_inundation(self, inundation: Inundation):
        inundation.integrate(
            ssc_boundary=self.platform.ssc,
            bulk_density=self.platform.bulk_density,
            settling_rate=self.platform.settling_rate,
        )
        result = inundation.summarize()
        result["n_cycles"] = 1
        if len(self._inundations) > 0 and self._inundations[-1].end == inundation.start:
            self.combine_inundations(result)
        else:
            self._inundations.append(result)
        elapsed = inundation.end - self.now
        self.platform.update(elapsed=elapsed, aggradation=inundation.aggradation)

    def combine_inundations(self, result):
        last = self._inundations[-1]
        last.n_cycles += 1
        last.end = result.end
        last.hydroperiod += result.hydroperiod
        last.aggradation += result.aggradation
        last.depth = (last.depth + result.depth) / last.n_cycles
        last.valid = last.valid and result.valid

    def run(self):
        with contextlib.suppress(KeyboardInterrupt):
            while self.now < self.end:
                self.step()

    # def equilibrate(self, target):
    #     with contextlib.suppress(KeyboardInterrupt):
    #         year = self.start.year
    #         delta = target
    #         while self.now < self.end and delta >= target:
    #             end = pd.to_datetime(year, utc=True).tz_convert(self.tz)
    #             init_elevation = self.platform.state["elevation"]
    #             self.run_until(end=end)
    #             delta = self.platform.state["elevation"] - init_elevation
    #         self.pbar.close()
    #         self.set_results()

    def close(self) -> None:
        self.set_results()
        self.pbar.close()
        self.logger.info("Simulation completed. Exiting.")

    def set_results(self, freq="A"):
        totals = normalize_timeseries(ts=self.platform.history, freq=self.freq)
        first = totals.iloc[0].to_frame().T
        totals = totals.resample(freq).last()
        first.index = [self.start]
        totals = pd.concat([first, totals])

        inundations = self._inundations.resample("A", on="start")[
            ["hydroperiod", "aggradation", "depth"]
        ].mean()

        tides = self.tides.summarize(freq=freq, start=totals.index[0], end=totals.index[-1])
        tides.index = tides.index + pd.DateOffset(months=-6)
        self.results = Bunch(totals=totals, inundations=inundations, tides=tides)

    # def reset(self):
    #     self.platform.reset()
    #     self.results = []
    #     self.n = 0
    #     self.now = self.start
    #     self.pbar.reset()


# @dataclass
# class Model:
#     # input these fields
#     water_levels: InitVar[pd.Series]
#     tides: Tides = field(init=False)
#     init_elevation: float
#     # params
#     ssc: float
#     grain_diameter: float
#     grain_density: float
#     bulk_density: float
#     organic_rate: float = 0.0
#     compaction_rate: float = 0.0
#     subsidence_rate: float = 0.0
#     amp_factor: float = 0.0
#     slr_rate: float = 0.0
#     solve_ivp_kwargs: dict = field(default_factory=dict)
#     tqdm_kwargs: dict = field(default_factory=dict)

#     # auto initialize these fields
#     settling_rate: float = field(init=False)
#     flood_cycles: list = field(default_factory=list)
#     inundations: list = field(default_factory=list)
#     platform: list = field(default_factory=list)
#     results: pd.DateOffset = field(init=False)
#     start: pd.Timestamp = field(init=False)
#     end: pd.Timestamp = field(init=False)
#     period: pd.Timedelta = field(init=False)
#     now: pd.Timestamp = field(init=False)
#     elevation: float = field(init=False)

#     def __post_init__(self, water_levels) -> None:
#         self.elevation = self.init_elevation
#         self.tides = Tides(water_levels=water_levels)
#         self.tides.amplify(factor=self.mtr / self.tides.summary.MN)
#         self.tides.raise_sea_level(slr=self.rslr_rate)

#         self.settling_rate = stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)

#         self.start = self.now = self.tides.start
#         self.end = self.tides.end
#         self.period = self.tides.period

#         self.istart = self.inow = 0
#         self.iend = len(self.tides.cycles) - 1

#         self.logger = logger.patch(lambda record: record["extra"].update(model_time=self.now))

#         result = {
#             "datetime": self.start,
#             "aggradation": 0.0,
#             "subsidence": 0.0,
#             "elevation": self.elevation,
#             "elevation_change": 0.0,
#         }
#         self.platform.append(result)

#         solve_ivp_kwargs = {
#             "method": "RK45",
#             "dense_output": False,
#             "first_step": None,
#             "max_step": np.inf,
#             "rtol": 1e-3,
#             "atol": 1e-6,
#         }
#         solve_ivp_kwargs |= self.solve_ivp_kwargs
#         self.solve_ivp_kwargs = solve_ivp_kwargs
#         self.setup_pbar()

#     @property
#     def rslr_rate(self):
#         return self.slr_rate + self.subsidence_rate + self.compaction_rate

#     @property
#     def elevation(self):
#         return self.elevation - self.calc_rate(rate=self.subsidence_rate)

#     def calc_rate(self, rate, t0=None, t1=None, freq=pd.to_timedelta("365.2425D")):
#         if t0 is None:
#             t0 = self.start
#         if t1 is None:
#             t1 = self.now
#         years = (t1 - t0) / freq
#         return rate * years

#     def setup_pbar(self, end_date=None, period=None):
#         if end_date:
#             self.end = pd.to_datetime(end_date)
#         if period:
#             self.end = self.start + pd.to_timedelta(period)
#         if self.period < pd.Timedelta("365.2425D"):
#             time_unit = pd.Timedelta("1D")
#             unit = "Day"
#         else:
#             time_unit = pd.Timedelta("365.2425D")
#             unit = "Year"
#         tqdm_opts = {
#             "total": round(self.period / time_unit, 2),
#             "leave": True,
#             "unit": unit,
#             "dynamic_ncols": True,
#             "maxinterval": 1,
#             "smoothing": 0,
#             "postfix": {"Elevation": f"{self.elevation:.3f} ({0.0:+.3f}) m"},
#         }
#         tqdm_opts |= self.tqdm_kwargs
#         self.tqdm_kwargs = tqdm_opts
#         self.pbar = tqdm(**self.tqdm_kwargs)
#         self.time_unit = time_unit

#     def close(self) -> None:
#         result = {
#             "datetime": self.end,
#             "aggradation": self.platform[-1]["aggradation"],
#             "subsidence": self.calc_rate(rate=self.subsidence_rate, t1=self.end),
#             "elevation": self.elevation,
#             "elevation_change": self.elevation - self.init_elevation,
#         }
#         self.platform.append(result)
#         self.compile_results()
#         self.pbar.close()
#         self.logger.info("Simulation completed. Exiting.")

#     def compile_results(self, freq="A"):
#         self.pbar.refresh()
#         platform = pd.DataFrame.from_records(data=self.platform, index="datetime")
#         platform = normalize_timeseries(ts=platform, freq=self.tides.water_levels.index.freq)
#         first = platform.iloc[0].to_frame().T
#         platform = platform.resample(freq).last()
#         first.index = [platform.index.shift(-1)[0]]
#         platform = pd.concat([first, platform])

#         inundations = pd.DataFrame.from_records(data=self.inundations).sort_values(by="start")
#         inundations = inundations.resample(freq, on="start").agg(
#             n_cycles=("n_cycles", "sum"),
#             aggradation=("aggradation", "sum"),
#             mean_hp=("hydroperiod", "mean"),
#             cum_hp=("hydroperiod", "sum"),
#             cum_depth=("cum_depth", "sum"),
#         )

#         subsidence = self.calc_rate(rate=self.subsidence_rate, t1=self.tides.water_levels.index)
#         tides = Tides(water_levels=self.tides.water_levels - subsidence)
#         tides = tides.summarize(freq=freq)
#         self.results = pd.concat([platform, inundations, tides], axis=1, keys=["platform", "inundations", "tides"])

#     def step(self):
#         if cycle := self.next_cycle():
#             self.process_cycle(cycle)
#         else:
#             self.now = self.end
#             self.inow = self.iend
#         self.update_pbar()

#     def update_pbar(self):
#         new_n = round((self.now - self.start) / self.time_unit, 2)
#         elevation_change = self.elevation - self.init_elevation
#         self.pbar.set_postfix({"Elevation": f"{self.elevation:.3f} ({elevation_change:+.3f}) m"}, refresh=False)
#         self.pbar.update(n=new_n - self.pbar.n)

#     def run(self):
#         with contextlib.suppress(KeyboardInterrupt):
#             while self.now < self.end:
#                 self.step()
#             self.close()

#     def next_cycle(self):
#         remaining = self.tides.cycles.loc[self.inow :]
#         above = remaining.high > self.elevation
#         if above.any():
#             i = above.idxmax()
#             return FloodCycle(
#                 cycle_id=i,
#                 water_levels=self.tides.make_cycle(i),
#                 init_rel_elevation=self.elevation,
#                 ssc_boundary=self.ssc,
#                 bulk_density=self.bulk_density,
#                 settling_rate=self.settling_rate,
#                 solve_ivp_opts=self.solve_ivp_kwargs,
#             )
#         else:
#             return None

#     def process_cycle(self, cycle):
#         cycle.integrate()
#         self.flood_cycles.append(cycle)
#         self.now = cycle.end
#         self.inow = cycle.cycle_id + 1
#         self.elevation += cycle.aggradation

#         if self.flooded:
#             last = self.inundations[-1]

#             self.inundations[-1].update(
#                 {
#                     "end": cycle.end,
#                     "n_cycles": last["n_cycles"] + 1,
#                     "aggradation": last["aggradation"] + cycle.aggradation,
#                     "hydroperiod": last["hydroperiod"] + cycle.hydroperiod,
#                     "cum_depth": last["cum_depth"] + cycle.slack.depth,
#                 }
#             )
#         else:
#             self.inundations.append(
#                 {
#                     "start": cycle.start,
#                     "end": cycle.end,
#                     "n_cycles": 1,
#                     "aggradation": cycle.aggradation,
#                     "hydroperiod": cycle.hydroperiod,
#                     "cum_depth": cycle.slack.depth,
#                 }
#             )

#         self.flooded = cycle.flooded

#         self.platform.append(
#             {
#                 "datetime": cycle.end,
#                 "aggradation": self.platform[-1]["aggradation"] + cycle.aggradation,
#                 "subsidence": self.calc_rate(rate=self.subsidence_rate, t1=cycle.end),
#                 "elevation": self.elevation,
#                 "elevation_change": self.elevation - self.init_elevation,
#             }
#         )

#     # def reset(self):
#     #     self.results = []
#     #     self.inundation_objects = []
#     #     self.invalid_inundations = []
#     #     self.inow = 0
#     #     self.now = self.start
#     #     self.rel_elevation = self.init_elevation
#     #     self.pbar.reset()
