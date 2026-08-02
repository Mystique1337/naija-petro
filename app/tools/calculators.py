"""Deterministic petroleum-engineering calculators.

Each calculator is a pure function returning numeric results, the formula used,
and units, so the assistant gives exact figures instead of estimates. They are
exposed three ways: the chat tool pre-pass, the /tools API, and (optionally) the
UI. Field units throughout unless noted.
"""
from __future__ import annotations

import math


def _round(x, n=4):
    try:
        if x == 0:
            return 0.0
        return round(x, max(0, n - int(math.floor(math.log10(abs(x))))))
    except Exception:
        return x


# --- plot series ------------------------------------------------------------
# Calculators whose natural output is a curve also return an optional "series"
# key so the UI can draw the chart without re-implementing any physics in
# JavaScript. Shape: {"x": [...], "y": [...], "x_label", "y_label", "title"}.
# It is strictly additive: degenerate inputs simply omit "series", never raise,
# and no existing key, formula, or signature changes.

_SERIES_POINTS = 41  # 40 intervals, dense enough to look smooth, small in JSON


def _linspace(a: float, b: float, n: int = _SERIES_POINTS) -> list[float]:
    """Evenly spaced points from a to b inclusive. Deterministic."""
    if n < 2:
        return [float(a)]
    step = (float(b) - float(a)) / (n - 1)
    return [float(a) + step * i for i in range(n)]


def _series(xs, ys, x_label: str, y_label: str, title: str):
    """JSON-safe plot series, or None if anything is degenerate or non-finite."""
    try:
        if not xs or not ys or len(xs) != len(ys):
            return None
        out_x: list[float] = []
        out_y: list[float] = []
        for x, y in zip(xs, ys):
            fx, fy = float(x), float(y)
            if not (math.isfinite(fx) and math.isfinite(fy)):
                return None
            out_x.append(round(fx, 6))
            out_y.append(round(fy, 6))
        return {"x": out_x, "y": out_y, "x_label": x_label, "y_label": y_label, "title": title}
    except Exception:
        return None


def arps_decline(qi: float, Di: float, t: float, b: float = 0.0) -> dict:
    """Arps decline. qi=initial rate, Di=nominal decline (per time unit), t=time, b: 0 exp, 1 harmonic, else hyperbolic."""
    qi, Di, t, b = float(qi), float(Di), float(t), float(b)
    if b == 0:
        q = qi * math.exp(-Di * t)
        Np = (qi - q) / Di if Di else 0.0
        formula = r"q = q_i e^{-D_i t}"
    elif b == 1:
        q = qi / (1 + Di * t)
        Np = (qi / Di) * math.log(qi / q) if Di and q else 0.0
        formula = r"q = q_i / (1 + D_i t)"
    else:
        q = qi / ((1 + b * Di * t) ** (1.0 / b))
        Np = (qi ** b / ((1 - b) * Di)) * (qi ** (1 - b) - q ** (1 - b)) if Di and b != 1 else 0.0
        formula = r"q = q_i / (1 + b D_i t)^{1/b}"
    out = {"rate_q": _round(q), "cumulative_Np": _round(Np), "formula": formula,
           "units": {"rate_q": "same as qi", "cumulative_Np": "qi x time"}}
    # Decline curve: rate against time, 0 to t (or ~3 time constants if t is 0).
    try:
        if Di > 0 and qi > 0:
            t_end = t if t > 0 else 3.0 / Di
            if t_end > 0:
                xs = _linspace(0.0, t_end)
                if b == 0:
                    ys = [qi * math.exp(-Di * x) for x in xs]
                    kind = "exponential"
                elif b == 1:
                    ys = [qi / (1 + Di * x) for x in xs]
                    kind = "harmonic"
                else:
                    ys = [qi / ((1 + b * Di * x) ** (1.0 / b)) for x in xs]
                    kind = f"hyperbolic, b={b:g}"
                s = _series(xs, ys, "Time t", "Rate q", f"Arps decline curve ({kind})")
                if s:
                    out["series"] = s
    except Exception:
        pass
    return out


def ooip_volumetric(area_acres: float, thickness_ft: float, porosity: float,
                    water_sat: float, bo: float) -> dict:
    """Original oil in place (STOIIP) by the volumetric method, in STB."""
    stoiip = 7758.0 * float(area_acres) * float(thickness_ft) * float(porosity) * (1 - float(water_sat)) / float(bo)
    return {"OOIP_STB": _round(stoiip), "OOIP_MMSTB": _round(stoiip / 1e6),
            "formula": r"N = 7758\,A\,h\,\phi\,(1-S_w)/B_o", "units": {"OOIP_STB": "STB"}}


def ogip_volumetric(area_acres: float, thickness_ft: float, porosity: float,
                    water_sat: float, bg_rcf_per_scf: float) -> dict:
    """Original gas in place (OGIP) by the volumetric method, in scf (Bg in reservoir cf per scf)."""
    ogip = 43560.0 * float(area_acres) * float(thickness_ft) * float(porosity) * (1 - float(water_sat)) / float(bg_rcf_per_scf)
    return {"OGIP_scf": _round(ogip), "OGIP_Bscf": _round(ogip / 1e9),
            "formula": r"G = 43560\,A\,h\,\phi\,(1-S_w)/B_g", "units": {"OGIP_scf": "scf"}}


def vogel_ipr(q_test: float, pwf_test: float, pr: float, pwf_target: float | None = None) -> dict:
    """Vogel IPR for saturated reservoirs. Returns qmax (AOF) and, if pwf_target given, the rate there."""
    q_test, pwf_test, pr = float(q_test), float(pwf_test), float(pr)
    x = pwf_test / pr
    qmax = q_test / (1 - 0.2 * x - 0.8 * x * x)
    out = {"qmax_AOF": _round(qmax),
           "formula": r"q/q_{max} = 1 - 0.2(P_{wf}/\bar P_r) - 0.8(P_{wf}/\bar P_r)^2",
           "units": {"qmax_AOF": "same as q_test"}}
    if pwf_target is not None:
        xt = float(pwf_target) / pr
        out["rate_at_pwf_target"] = _round(qmax * (1 - 0.2 * xt - 0.8 * xt * xt))
    # IPR curve: rate against flowing bottomhole pressure, 0 to reservoir pressure.
    try:
        if pr > 0 and math.isfinite(qmax) and qmax > 0:
            xs = _linspace(0.0, pr)
            ys = [qmax * (1 - 0.2 * (p / pr) - 0.8 * (p / pr) ** 2) for p in xs]
            s = _series(xs, ys, "Flowing bottomhole pressure Pwf (psi)", "Rate q",
                        "Vogel IPR curve")
            if s:
                out["series"] = s
    except Exception:
        pass
    return out


def darcy_radial_oil(k_md: float, h_ft: float, pr_psi: float, pwf_psi: float,
                     mu_cp: float, bo: float, re_ft: float, rw_ft: float, skin: float = 0.0) -> dict:
    """Pseudo-steady-state radial oil inflow (field units), STB/day."""
    num = 0.00708 * float(k_md) * float(h_ft) * (float(pr_psi) - float(pwf_psi))
    den = float(mu_cp) * float(bo) * (math.log(float(re_ft) / float(rw_ft)) - 0.75 + float(skin))
    q = num / den if den else 0.0
    return {"rate_STB_per_day": _round(q),
            "formula": r"q = \frac{0.00708\,k\,h\,(\bar P_r - P_{wf})}{\mu\,B_o\,(\ln(r_e/r_w)-0.75+s)}",
            "units": {"rate_STB_per_day": "STB/day"}}


def hydrostatic_pressure(mud_weight_ppg: float, tvd_ft: float) -> dict:
    """Hydrostatic (mud column) pressure and gradient, field units."""
    p = 0.052 * float(mud_weight_ppg) * float(tvd_ft)
    return {"pressure_psi": _round(p), "gradient_psi_per_ft": _round(0.052 * float(mud_weight_ppg)),
            "formula": r"P = 0.052 \times MW(\text{ppg}) \times TVD(\text{ft})", "units": {"pressure_psi": "psi"}}


def standing_pb(rs_scf_stb: float, gas_grav: float, api: float, temp_f: float) -> dict:
    """Standing bubble-point pressure (psia). Rs in scf/STB, gas_grav (air=1), oil API, temp degF."""
    rs, g, api, t = float(rs_scf_stb), float(gas_grav), float(api), float(temp_f)
    pb = 18.2 * ((rs / g) ** 0.83 * 10 ** (0.00091 * t - 0.0125 * api) - 1.4)
    return {"bubble_point_psia": _round(pb),
            "formula": r"P_b = 18.2\left[(R_s/\gamma_g)^{0.83}\,10^{0.00091T-0.0125\,API} - 1.4\right]",
            "units": {"bubble_point_psia": "psia"}}


def standing_bo(rs_scf_stb: float, gas_grav: float, api: float, temp_f: float) -> dict:
    """Standing oil formation volume factor Bo (RB/STB) at/below bubble point."""
    rs, g, api, t = float(rs_scf_stb), float(gas_grav), float(api), float(temp_f)
    oil_grav = 141.5 / (131.5 + api)
    bo = 0.9759 + 0.00012 * (rs * (g / oil_grav) ** 0.5 + 1.25 * t) ** 1.2
    return {"Bo_RB_per_STB": _round(bo), "oil_specific_gravity": _round(oil_grav),
            "formula": r"B_o = 0.9759 + 0.00012\,[R_s(\gamma_g/\gamma_o)^{0.5} + 1.25T]^{1.2}",
            "units": {"Bo_RB_per_STB": "RB/STB"}}


def gas_fvf(z: float, temp_f: float, pressure_psia: float) -> dict:
    """Gas formation volume factor Bg (reservoir cf per scf). z=compressibility factor."""
    z, t, p = float(z), float(temp_f), float(pressure_psia)
    bg = 0.02827 * z * (t + 460.0) / p
    return {"Bg_rcf_per_scf": _round(bg), "Bg_rb_per_mscf": _round(bg * 1000 / 5.615),
            "formula": r"B_g = 0.02827\,zT/P \;(T\ \text{in}\ ^\circ R)", "units": {"Bg_rcf_per_scf": "rcf/scf"}}


def productivity_index(q_test: float, pr_psi: float, pwf_test: float, pwf_target: float | None = None) -> dict:
    """Straight-line (undersaturated) IPR. J = q/(Pr - Pwf)."""
    q, pr, pwf = float(q_test), float(pr_psi), float(pwf_test)
    J = q / (pr - pwf) if pr != pwf else 0.0
    out = {"J_STB_per_day_psi": _round(J), "AOF_at_pwf_0": _round(J * pr),
           "formula": r"J = q/(\bar P_r - P_{wf});\ q = J(\bar P_r - P_{wf})", "units": {"J_STB_per_day_psi": "STB/d/psi"}}
    if pwf_target is not None:
        out["rate_at_pwf_target"] = _round(J * (pr - float(pwf_target)))
    # Straight-line IPR: rate against flowing bottomhole pressure, 0 to Pr.
    try:
        if pr > 0 and math.isfinite(J) and J > 0:
            xs = _linspace(0.0, pr)
            ys = [J * (pr - p) for p in xs]
            s = _series(xs, ys, "Flowing bottomhole pressure Pwf (psi)", "Rate q (STB/day)",
                        "Straight-line IPR")
            if s:
                out["series"] = s
    except Exception:
        pass
    return out


def arps_eur_exponential(qi: float, D_per_year: float, q_abandon: float) -> dict:
    """Exponential decline EUR to an economic limit rate. qi, q_abandon in same rate units, D per year."""
    qi, D, qa = float(qi), float(D_per_year), float(q_abandon)
    eur = (qi - qa) / D if D else 0.0
    t = math.log(qi / qa) / D if (D and qa) else 0.0
    out = {"EUR": _round(eur), "years_to_limit": _round(t),
           "formula": r"EUR = (q_i - q_a)/D;\ t = \ln(q_i/q_a)/D",
           "units": {"EUR": "rate x year", "years_to_limit": "years"}}
    # Decline profile: rate against time, initial rate down to the abandonment rate.
    try:
        if D > 0 and qi > 0 and qa > 0 and qi > qa:
            t_end = math.log(qi / qa) / D
            if t_end > 0:
                xs = _linspace(0.0, t_end)
                ys = [qi * math.exp(-D * x) for x in xs]
                s = _series(xs, ys, "Time (years)", "Rate q",
                            "Exponential decline to the abandonment rate")
                if s:
                    out["series"] = s
    except Exception:
        pass
    return out


def recovery_factor(np_stb: float, ooip_stb: float) -> dict:
    """Recovery factor = cumulative production / original oil in place."""
    rf = float(np_stb) / float(ooip_stb) if float(ooip_stb) else 0.0
    return {"recovery_factor": _round(rf), "recovery_percent": _round(rf * 100),
            "formula": r"RF = N_p/N", "units": {"recovery_percent": "%"}}


# name -> (function, human description used in the tool-selection prompt)
CALCULATORS: dict[str, tuple] = {
    "arps_decline": (arps_decline,
        "Production decline (Arps). args: qi (initial rate), Di (nominal decline per time unit), t (elapsed time, same units as Di), b (0=exponential, 1=harmonic, 0<b<1=hyperbolic)."),
    "ooip_volumetric": (ooip_volumetric,
        "Original oil in place / STOIIP (volumetric). args: area_acres, thickness_ft, porosity (fraction 0-1), water_sat (fraction 0-1), bo (RB/STB)."),
    "ogip_volumetric": (ogip_volumetric,
        "Original gas in place / OGIP (volumetric). args: area_acres, thickness_ft, porosity (fraction), water_sat (fraction), bg_rcf_per_scf (reservoir cf per scf)."),
    "vogel_ipr": (vogel_ipr,
        "Vogel inflow performance (saturated). args: q_test, pwf_test, pr (avg reservoir pressure), optional pwf_target."),
    "darcy_radial_oil": (darcy_radial_oil,
        "Radial oil inflow rate, pseudo-steady-state, field units. args: k_md, h_ft, pr_psi, pwf_psi, mu_cp, bo, re_ft, rw_ft, optional skin."),
    "hydrostatic_pressure": (hydrostatic_pressure,
        "Hydrostatic/mud-column pressure. args: mud_weight_ppg, tvd_ft."),
    "standing_pb": (standing_pb,
        "Standing bubble-point pressure. args: rs_scf_stb, gas_grav (air=1), api, temp_f."),
    "standing_bo": (standing_bo,
        "Standing oil formation volume factor Bo. args: rs_scf_stb, gas_grav, api, temp_f."),
    "gas_fvf": (gas_fvf,
        "Gas formation volume factor Bg. args: z (compressibility factor), temp_f, pressure_psia."),
    "productivity_index": (productivity_index,
        "Straight-line (undersaturated) IPR / productivity index. args: q_test, pr_psi, pwf_test, optional pwf_target."),
    "arps_eur_exponential": (arps_eur_exponential,
        "Exponential decline EUR to an economic limit. args: qi, D_per_year, q_abandon."),
    "recovery_factor": (recovery_factor,
        "Recovery factor. args: np_stb (cumulative), ooip_stb (original oil in place)."),
}

TOOL_MENU = "\n".join(f"- {name}: {desc}" for name, (_, desc) in CALCULATORS.items())

# UI-facing argument specs: ordered fields with friendly labels for the manual
# calculator picker. `opt` marks an optional argument. Ordered for the UI
# (in-place, decline, IPR/inflow, drilling, PVT).
TOOL_SPECS: dict[str, dict] = {
    "ooip_volumetric": {"label": "Oil in place (STOIIP, volumetric)", "args": [
        {"name": "area_acres", "label": "Drainage area (acres)"},
        {"name": "thickness_ft", "label": "Net pay thickness (ft)"},
        {"name": "porosity", "label": "Porosity (fraction 0-1)"},
        {"name": "water_sat", "label": "Water saturation (fraction 0-1)"},
        {"name": "bo", "label": "Oil FVF Bo (RB/STB)"},
    ]},
    "ogip_volumetric": {"label": "Gas in place (OGIP, volumetric)", "args": [
        {"name": "area_acres", "label": "Drainage area (acres)"},
        {"name": "thickness_ft", "label": "Net pay thickness (ft)"},
        {"name": "porosity", "label": "Porosity (fraction 0-1)"},
        {"name": "water_sat", "label": "Water saturation (fraction 0-1)"},
        {"name": "bg_rcf_per_scf", "label": "Gas FVF Bg (rcf/scf)"},
    ]},
    "recovery_factor": {"label": "Recovery factor", "args": [
        {"name": "np_stb", "label": "Cumulative produced Np (STB)"},
        {"name": "ooip_stb", "label": "Original oil in place N (STB)"},
    ]},
    "arps_decline": {"label": "Arps production decline", "args": [
        {"name": "qi", "label": "Initial rate qi"},
        {"name": "Di", "label": "Nominal decline Di (per time unit)"},
        {"name": "t", "label": "Elapsed time t"},
        {"name": "b", "label": "b exponent (0=exp, 1=harmonic)", "opt": True},
    ]},
    "arps_eur_exponential": {"label": "EUR (exponential decline)", "args": [
        {"name": "qi", "label": "Initial rate qi"},
        {"name": "D_per_year", "label": "Decline D (per year)"},
        {"name": "q_abandon", "label": "Abandonment rate"},
    ]},
    "vogel_ipr": {"label": "Vogel IPR (saturated)", "args": [
        {"name": "q_test", "label": "Test rate q"},
        {"name": "pwf_test", "label": "Test flowing BHP Pwf"},
        {"name": "pr", "label": "Avg reservoir pressure Pr"},
        {"name": "pwf_target", "label": "Target Pwf", "opt": True},
    ]},
    "productivity_index": {"label": "Productivity index (straight-line IPR)", "args": [
        {"name": "q_test", "label": "Test rate q"},
        {"name": "pr_psi", "label": "Avg reservoir pressure Pr (psi)"},
        {"name": "pwf_test", "label": "Test flowing BHP Pwf (psi)"},
        {"name": "pwf_target", "label": "Target Pwf (psi)", "opt": True},
    ]},
    "darcy_radial_oil": {"label": "Radial oil inflow (Darcy, PSS)", "args": [
        {"name": "k_md", "label": "Permeability k (md)"},
        {"name": "h_ft", "label": "Net pay h (ft)"},
        {"name": "pr_psi", "label": "Reservoir pressure Pr (psi)"},
        {"name": "pwf_psi", "label": "Flowing BHP Pwf (psi)"},
        {"name": "mu_cp", "label": "Oil viscosity (cp)"},
        {"name": "bo", "label": "Oil FVF Bo (RB/STB)"},
        {"name": "re_ft", "label": "Drainage radius re (ft)"},
        {"name": "rw_ft", "label": "Wellbore radius rw (ft)"},
        {"name": "skin", "label": "Skin", "opt": True},
    ]},
    "hydrostatic_pressure": {"label": "Hydrostatic (mud column) pressure", "args": [
        {"name": "mud_weight_ppg", "label": "Mud weight (ppg)"},
        {"name": "tvd_ft", "label": "True vertical depth (ft)"},
    ]},
    "standing_pb": {"label": "Bubble-point pressure (Standing)", "args": [
        {"name": "rs_scf_stb", "label": "Solution GOR Rs (scf/STB)"},
        {"name": "gas_grav", "label": "Gas gravity (air=1)"},
        {"name": "api", "label": "Oil gravity (API)"},
        {"name": "temp_f", "label": "Temperature (deg F)"},
    ]},
    "standing_bo": {"label": "Oil FVF Bo (Standing)", "args": [
        {"name": "rs_scf_stb", "label": "Solution GOR Rs (scf/STB)"},
        {"name": "gas_grav", "label": "Gas gravity (air=1)"},
        {"name": "api", "label": "Oil gravity (API)"},
        {"name": "temp_f", "label": "Temperature (deg F)"},
    ]},
    "gas_fvf": {"label": "Gas FVF Bg", "args": [
        {"name": "z", "label": "z-factor"},
        {"name": "temp_f", "label": "Temperature (deg F)"},
        {"name": "pressure_psia", "label": "Pressure (psia)"},
    ]},
}

_KEYWORDS = (
    "calculate", "compute", "decline", "arps", "ipr", "vogel", "aof", "ooip", "stoiip",
    "ogip", "oip", "darcy", "inflow", "rate", "hydrostatic", "mud weight", "pressure",
    "eur", "qmax", "flow rate", "volumetric", "in place", "psi", "bopd", "porosity",
)


def needs_tools(query: str) -> bool:
    """Cheap gate: does the query look like it wants a numeric calculation?"""
    t = (query or "").lower()
    has_digit = any(c.isdigit() for c in t)
    return has_digit and any(k in t for k in _KEYWORDS)


def run_tool(name: str, args: dict) -> dict:
    if name not in CALCULATORS:
        return {"error": f"unknown calculator '{name}'"}
    fn, _ = CALCULATORS[name]
    try:
        clean = {k: v for k, v in (args or {}).items() if v is not None}
        return fn(**clean)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
