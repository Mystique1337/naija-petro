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
    return out


def arps_eur_exponential(qi: float, D_per_year: float, q_abandon: float) -> dict:
    """Exponential decline EUR to an economic limit rate. qi, q_abandon in same rate units, D per year."""
    qi, D, qa = float(qi), float(D_per_year), float(q_abandon)
    eur = (qi - qa) / D if D else 0.0
    t = math.log(qi / qa) / D if (D and qa) else 0.0
    return {"EUR": _round(eur), "years_to_limit": _round(t),
            "formula": r"EUR = (q_i - q_a)/D;\ t = \ln(q_i/q_a)/D", "units": {"EUR": "rate x year", "years_to_limit": "years"}}


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
