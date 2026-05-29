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
    return {"rate_q": _round(q), "cumulative_Np": _round(Np), "formula": formula,
            "units": {"rate_q": "same as qi", "cumulative_Np": "qi x time"}}


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
}

TOOL_MENU = "\n".join(f"- {name}: {desc}" for name, (_, desc) in CALCULATORS.items())

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
