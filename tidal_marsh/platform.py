from dataclasses import dataclass, field, InitVar
import pandas as pd
from sklearn.utils import Bunch
from .utils import stokes_settling
from .inundation import Inundation


@dataclass
class Platform:
    init_elevation: float
    ssc: float
    grain_diameter: float
    bulk_density: float
    grain_density: float = 2650.0
    organic_rate: float = 0.0
    compaction_rate: float = 0.0
    subsidence_rate: float = 0.0
    settling_rate: float = field(init=False, default=None)
    totals: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.settling_rate = stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)
        data = {
            "elapsed": pd.Timedelta("0D"),
            "aggradation": 0.0,
            "subsidence": 0.0,
            "net": 0.0,
            "elevation": self.init_elevation,
        }
        self.totals.append(data)

    def reset(self):
        self.totals = []
        self.inundations = []
        self.__post_init__()

    @property
    def state(self):
        return self.totals[-1]

    @property
    def surface(self):
        return self.init_elevation + self.totals[-1]["aggradation"]

    def calc_subsidence(self, elapsed: pd.Timedelta = None):
        return self.subsidence_rate * elapsed / pd.to_timedelta("365.2425D")

    @property
    def history(self):
        return pd.DataFrame.from_records(self.totals, index="elapsed")
        # return Bunch(
        #     totals=pd.DataFrame.from_records(self.totals, index="elapsed"),
        #     inundations=pd.concat(self.inundations, axis=1).T,
        # )

    # def inundate(self, inundation: Inundation):
    #     self.flood_cycles.append(inundation)
    #     inundation.integrate(
    #         ssc_boundary=self.ssc,
    #         bulk_density=self.bulk_density,
    #         settling_rate=self.settling_rate,
    #     )
    #     self.process_inundation(inundation)

    # def process_inundation(self, inundation: Inundation):
    #     result = inundation.summarize()
    #     result["n_cycles"] = 1
    #     if len(self.inundations) > 0 and self.inundations[-1].end == inundation.start:
    #         self.combine_results(result)
    #     else:
    #         self.inundations.append(result)
    #     aggradation = self.state["aggradation"] + inundation.aggradation
    #     subsidence = self.calc_subsidence(time=inundation.end)
    #     self.update(inundation.end, aggradation, subsidence)

    def update(self, elapsed, aggradation=None):
        if aggradation is None:
            aggradation = self.state["aggradation"]
        else:
            aggradation = self.state["aggradation"] + aggradation
        subsidence = self.state['subsidence'] + self.calc_subsidence(elapsed=elapsed)
        net = aggradation - subsidence
        totals = {
            "elapsed": self.state['elapsed'] + elapsed,
            "aggradation": aggradation,
            "subsidence": subsidence,
            "net": net,
            "elevation": self.init_elevation + net,
        }
        self.totals.append(totals)
