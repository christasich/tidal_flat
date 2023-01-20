from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Platform:
    time_ref: pd.Timestamp
    elevation_ref: float = 0.0
    elapsed: pd.Timedelta = field(init=False, default=pd.Timedelta(0))
    aggradation: float = field(init=False, default=0.0)
    records: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.log()

    @property
    def surface(self) -> float:
        return self.aggradation + self.elevation_ref

    def update(self, time, aggradation) -> None:
        self.elapsed += time
        self.aggradation += aggradation
        self.log()

    def log(self) -> None:
        record = {"datetime": self.elapsed, "aggradation": self.aggradation}
        self.records.append(record)

    def report(self) -> pd.Series:
        s = pd.DataFrame.from_records(data=self.records, index="datetime").squeeze(axis=1)
        s.index += self.time_ref
        return s
