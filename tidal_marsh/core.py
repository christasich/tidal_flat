from functools import cache
from pathlib import Path
import pandas as pd

from scipy.misc import derivative

from .model import Model
from .platform import Platform
from .tides import Tides


# def load_tides(tides, start=None, end=None):
#     if isinstance(tides, (str, Path)):
#         return Tides(pd.read_pickle(tides).loc[start:end])
#     elif isinstance(tides, pd.Series):
#         return Tides(tides.loc[start:end])


def get_results(model) -> pd.DataFrame:
    platform = model.results.totals.elevation.resample("A").mean().to_frame("avg_elev")
    platform["start_elev"] = model.results.totals.elevation.resample("A").first()
    platform["end_elev"] = model.results.totals.elevation.resample("A").last()
    platform["total_agg"] = model.results.totals.aggradation.diff().fillna(0).resample("A").sum()
    platform["total_sub"] = model.results.totals.subsidence.diff().fillna(0).resample("A").sum()

    inundations = model.results.inundations.resample("A", on="start").agg(
        avg_hp=("hydroperiod", "mean"),
        avg_agg=("aggradation", "mean"),
        avg_depth=("max_depth", "mean"),
        total=("num_cycles", "sum"),
    )

    tides = model.tides.summarize("A")

    results = pd.concat([platform, inundations, tides], keys=["platform", "inundations", "tides"], axis=1).rename_axis(
        index="datetime"
    )
    results[("params", "ssc")] = model.ssc
    results[("params", "bulk_density")] = model.bulk_density
    results[("params", "grain_diameter")] = model.grain_diameter
    results[("params", "grain_density")] = model.grain_density
    results[("params", "settling_rate")] = model.settling_rate
    results[("params", "deep_sub")] = model.deep_sub
    results[("params", "org_sed")] = model.org_sed
    results[("params", "compaction")] = model.compaction

    return results


def simulate_platform(
    init_elevation,
    water_levels,
    ssc,
    bulk_density,
    grain_diameter,
    grain_density=2.65e3,
    slr=0.0,
    deep_sub=0.0,
    disable_pbar=False,
    compile_results=True,
):
    # tides = load_tides(water_levels)
    tides = Tides(water_levels)
    if mtr is not None:
        af = mtr / tides.datums.MN
        tides = tides.amplify(factor=af)
    else:
        mtr = tides.datums.MN
        af = 1.0
    if slr != 0.0:
        tides = tides.raise_sea_level(slr=slr)

    platform = Platform(time_ref=tides.start, elevation_ref=init_elevation)

    rslr = slr + -deep_sub
    pbar_opts = {
        "desc": (
            f"TR={mtr:.2f} m ({af - 1:+.0%}) | SSC={ssc:.3f} g/L | RSLR={(rslr)*1000:+.1f} mm |"
            f" N0={init_elevation:+.2f} m"
        ),
        "leave": False,
        "disable": disable_pbar,
    }

    model = Model(
        ssc=ssc,
        grain_diameter=grain_diameter,
        grain_density=grain_density,
        bulk_density=bulk_density,
        deep_sub=deep_sub,
        tides=tides,
        platform=platform,
        pbar_opts=pbar_opts,
    )
    model.run()
    model.close(compile_results=compile_results)
    return model


def find_equilibrium(tides, target, params, lower_elev, upper_elev, margin=1e-6) -> float:
    @cache
    def change_per_year(init_elevation) -> float:
        model = Model(
            ssc=params.ssc,
            grain_diameter=params.grain_diameter,
            grain_density=params.grain_density,
            bulk_density=params.bulk_density,
            tides=tides,
            platform=Platform(time_ref=tides.start, elevation_ref=init_elevation),
            pbar_opts={"disable": True},
        )
        model.run()
        return model.change_per_year

    try:
        eq_elev = illinois_algorithm(change_per_year, a=upper_elev, b=lower_elev, y=target, margin=margin)
        return eq_elev
    except AssertionError as e:
        if "f(a) must be greater than lower bound" in str(e):
            raise ValueError(f"Upper bound is too low. {upper_elev} is lower than equilibrium elevation.")
        elif "f(b) must be less than upper bound" in str(e):
            raise ValueError(f"Lower bound is too high. {lower_elev} is higher than equilibrium elevation.")

    # eq = {
    #     "start": start,
    #     "end": end,
    #     "period": end - start,
    #     "ssc": params["ssc"],
    #     "mtr": params["mtr"],
    #     "slr": params["slr"],
    # }

    # try:
    #     root = illinois_algorithm(change_per_year, a=upper_elev, b=lower_elev, y=params["slr"], margin=margin)
    #     eq.update({"eq_elev": root, "error": None})
    #     return eq
    # except Exception as e:
    #     eq.update({"eq_elev": None, "error": e})
    #     return eq


def illinois_algorithm(f, a, b, y, margin=1e-6):
    """Bracketed approach of Root-finding with illinois method
    Parameters
    ----------
    f: callable, continuous function
    a: float, lower bound to be searched
    b: float, upper bound to be searched
    y: float, target value
    margin: float, margin of error in absolute term
    Returns
    -------
    A float c, where f(c) is within the margin of y
    """

    assert y >= (lower := f(a)), f"f(a) must be greater than lower bound. {y} < {lower}"
    assert y <= (upper := f(b)), f"f(b) must be less than upper bound. {y} > {upper}"

    stagnant = 0

    n = 0
    while True:
        c = ((a * (upper - y)) - (b * (lower - y))) / (upper - lower)
        if abs((y_c := f(c)) - y) < margin:
            # found!
            return c
        elif y < y_c:
            b, upper = c, y_c
            if stagnant == -1:
                # Lower bound is stagnant!
                lower += (y - lower) / 2
            stagnant = -1
        else:
            a, lower = c, y_c
            if stagnant == 1:
                # Upper bound is stagnant!
                upper -= (upper - y) / 2
            stagnant = 1
        n += 1
        if n > 500:
            raise RuntimeError(
                f"Illinois algorithm failed to converge. Upper is {upper}, lower is {lower}, f(c) is {y_c}, error is"
                f" {abs(y_c - y)}, margin is {margin}."
            )
    # raise RuntimeError(
    #     f"Illinois algorithm failed to converge. Upper is {upper}, lower is {lower}, f(c) is {y_c}, error is"
    #     f" {abs(y_c - y)}, margin is {margin}."
    # )
