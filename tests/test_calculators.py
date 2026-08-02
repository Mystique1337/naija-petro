"""Numeric checks for the deterministic petroleum calculators.

Every expected value below was worked out by hand from the formula in the
docstring of the function under test, not by running the function. The module
rounds results to roughly five significant figures (`_round`), so a relative
tolerance of 1e-3 is comfortably tight enough to catch a wrong formula while
ignoring that rounding.
"""
from __future__ import annotations

import inspect

import pytest

from app.tools import calculators as calc

REL = 1e-3


# --------------------------------------------------------------------------- #
# Volumetrics
# --------------------------------------------------------------------------- #
def test_ooip_volumetric():
    # N = 7758 x 640 x 50 x 0.20 x (1 - 0.30) / 1.2 = 28,963,200 STB
    out = calc.ooip_volumetric(area_acres=640, thickness_ft=50, porosity=0.20,
                               water_sat=0.30, bo=1.2)
    assert out["OOIP_STB"] == pytest.approx(28_963_200.0, rel=REL)
    assert out["OOIP_MMSTB"] == pytest.approx(28.9632, rel=REL)
    assert out["units"]["OOIP_STB"] == "STB"


def test_ogip_volumetric():
    # G = 43560 x 640 x 50 x 0.20 x 0.70 / 0.005 = 39,029,760,000 scf
    out = calc.ogip_volumetric(area_acres=640, thickness_ft=50, porosity=0.20,
                               water_sat=0.30, bg_rcf_per_scf=0.005)
    assert out["OGIP_scf"] == pytest.approx(39_029_760_000.0, rel=REL)
    assert out["OGIP_Bscf"] == pytest.approx(39.02976, rel=REL)


def test_recovery_factor_is_a_plain_ratio():
    out = calc.recovery_factor(np_stb=2_000_000, ooip_stb=10_000_000)
    assert out["recovery_factor"] == pytest.approx(0.2, rel=REL)
    assert out["recovery_percent"] == pytest.approx(20.0, rel=REL)


def test_recovery_factor_zero_ooip_is_guarded():
    out = calc.recovery_factor(np_stb=1000, ooip_stb=0)
    assert out["recovery_factor"] == 0.0
    assert out["recovery_percent"] == 0.0


# --------------------------------------------------------------------------- #
# Decline
# --------------------------------------------------------------------------- #
def test_arps_decline_exponential():
    # b = 0: q = 1000 e^(-0.1 x 5) = 1000 x 0.60653066 = 606.53066
    #        Np = (1000 - 606.53066) / 0.1 = 3934.6934
    out = calc.arps_decline(qi=1000, Di=0.1, t=5, b=0)
    assert out["rate_q"] == pytest.approx(606.53066, rel=REL)
    assert out["cumulative_Np"] == pytest.approx(3934.6934, rel=REL)
    assert "e^{-D_i t}" in out["formula"]


def test_arps_decline_harmonic():
    # b = 1: q = 1000 / (1 + 0.1 x 5) = 666.6667
    #        Np = (1000/0.1) ln(1000/666.6667) = 10000 ln(1.5) = 4054.651
    out = calc.arps_decline(qi=1000, Di=0.1, t=5, b=1)
    assert out["rate_q"] == pytest.approx(666.6667, rel=REL)
    assert out["cumulative_Np"] == pytest.approx(4054.651, rel=REL)
    assert out["formula"] == r"q = q_i / (1 + D_i t)"


def test_arps_decline_hyperbolic():
    # b = 0.5: (1 + 0.5 x 0.1 x 5) = 1.25, q = 1000 / 1.25^2 = 640
    #          Np = (sqrt(1000)/(0.5 x 0.1)) (sqrt(1000) - sqrt(640))
    #             = (1000 - sqrt(640000)) / 0.05 = (1000 - 800) / 0.05 = 4000
    out = calc.arps_decline(qi=1000, Di=0.1, t=5, b=0.5)
    assert out["rate_q"] == pytest.approx(640.0, rel=REL)
    assert out["cumulative_Np"] == pytest.approx(4000.0, rel=REL)
    assert "(1 + b D_i t)^{1/b}" in out["formula"]


def test_arps_decline_defaults_to_exponential():
    assert calc.arps_decline(1000, 0.1, 5)["rate_q"] == calc.arps_decline(1000, 0.1, 5, 0)["rate_q"]


def test_arps_eur_exponential():
    # EUR = (1000 - 100) / 0.2 = 4500;  t = ln(10) / 0.2 = 11.5129
    out = calc.arps_eur_exponential(qi=1000, D_per_year=0.2, q_abandon=100)
    assert out["EUR"] == pytest.approx(4500.0, rel=REL)
    assert out["years_to_limit"] == pytest.approx(11.5129, rel=REL)


def test_arps_eur_zero_decline_is_guarded():
    out = calc.arps_eur_exponential(qi=1000, D_per_year=0, q_abandon=100)
    assert out["EUR"] == 0.0
    assert out["years_to_limit"] == 0.0


# --------------------------------------------------------------------------- #
# Inflow performance
# --------------------------------------------------------------------------- #
def test_vogel_ipr_without_target():
    # x = 2000/3000 = 2/3; 1 - 0.2(2/3) - 0.8(4/9) = 23/45
    # qmax = 500 x 45/23 = 978.2609
    out = calc.vogel_ipr(q_test=500, pwf_test=2000, pr=3000)
    assert out["qmax_AOF"] == pytest.approx(978.2609, rel=REL)
    assert "rate_at_pwf_target" not in out


def test_vogel_ipr_with_target():
    # xt = 0.5 -> 1 - 0.1 - 0.2 = 0.7 -> 978.2609 x 0.7 = 684.7826
    out = calc.vogel_ipr(q_test=500, pwf_test=2000, pr=3000, pwf_target=1500)
    assert out["rate_at_pwf_target"] == pytest.approx(684.7826, rel=REL)


def test_vogel_ipr_target_zero_returns_qmax():
    out = calc.vogel_ipr(q_test=500, pwf_test=2000, pr=3000, pwf_target=0)
    assert out["rate_at_pwf_target"] == pytest.approx(out["qmax_AOF"], rel=REL)


def test_productivity_index():
    # J = 500 / (3000 - 2500) = 1.0 STB/d/psi; AOF at Pwf = 0 is J x Pr = 3000
    out = calc.productivity_index(q_test=500, pr_psi=3000, pwf_test=2500)
    assert out["J_STB_per_day_psi"] == pytest.approx(1.0, rel=REL)
    assert out["AOF_at_pwf_0"] == pytest.approx(3000.0, rel=REL)
    assert "rate_at_pwf_target" not in out


def test_productivity_index_with_target():
    out = calc.productivity_index(q_test=500, pr_psi=3000, pwf_test=2500, pwf_target=2000)
    assert out["rate_at_pwf_target"] == pytest.approx(1000.0, rel=REL)


def test_productivity_index_no_drawdown_does_not_divide_by_zero():
    out = calc.productivity_index(q_test=500, pr_psi=3000, pwf_test=3000, pwf_target=1500)
    assert out["J_STB_per_day_psi"] == 0.0
    assert out["AOF_at_pwf_0"] == 0.0
    assert out["rate_at_pwf_target"] == 0.0


def test_darcy_radial_oil():
    # q = 0.00708 x 100 x 50 x (3000 - 2000)
    #     / [1.2 x 1.25 x (ln(1000/0.5) - 0.75 + 0)]
    #   = 35400 / (1.5 x 6.8509025) = 3444.80
    out = calc.darcy_radial_oil(k_md=100, h_ft=50, pr_psi=3000, pwf_psi=2000,
                                mu_cp=1.2, bo=1.25, re_ft=1000, rw_ft=0.5)
    assert out["rate_STB_per_day"] == pytest.approx(3444.80, rel=REL)
    assert out["units"]["rate_STB_per_day"] == "STB/day"


def test_darcy_radial_oil_positive_skin_reduces_rate():
    base = calc.darcy_radial_oil(100, 50, 3000, 2000, 1.2, 1.25, 1000, 0.5)
    damaged = calc.darcy_radial_oil(100, 50, 3000, 2000, 1.2, 1.25, 1000, 0.5, skin=5)
    assert damaged["rate_STB_per_day"] < base["rate_STB_per_day"]


# --------------------------------------------------------------------------- #
# Drilling + PVT
# --------------------------------------------------------------------------- #
def test_hydrostatic_pressure():
    # P = 0.052 x 10 ppg x 10000 ft = 5200 psi, gradient 0.52 psi/ft
    out = calc.hydrostatic_pressure(mud_weight_ppg=10, tvd_ft=10000)
    assert out["pressure_psi"] == pytest.approx(5200.0, rel=REL)
    assert out["gradient_psi_per_ft"] == pytest.approx(0.52, rel=REL)
    assert out["units"]["pressure_psi"] == "psi"


def test_hydrostatic_pressure_scales_linearly():
    shallow = calc.hydrostatic_pressure(12, 5000)["pressure_psi"]
    deep = calc.hydrostatic_pressure(12, 10000)["pressure_psi"]
    assert deep == pytest.approx(2 * shallow, rel=REL)


def test_standing_pb():
    # Pb = 18.2 [ (500/0.8)^0.83 x 10^(0.00091 x 180 - 0.0125 x 35) - 1.4 ]
    #    = 18.2 [ 625^0.83 x 10^(-0.2737) - 1.4 ] = 2001.98 psia
    out = calc.standing_pb(rs_scf_stb=500, gas_grav=0.8, api=35, temp_f=180)
    assert out["bubble_point_psia"] == pytest.approx(2001.98, rel=REL)
    assert out["units"]["bubble_point_psia"] == "psia"


def test_standing_bo():
    # gamma_o = 141.5 / (131.5 + 35) = 0.849850
    # Bo = 0.9759 + 0.00012 [500 (0.8/0.849850)^0.5 + 1.25 x 180]^1.2 = 1.29269
    out = calc.standing_bo(rs_scf_stb=500, gas_grav=0.8, api=35, temp_f=180)
    assert out["oil_specific_gravity"] == pytest.approx(0.849850, rel=REL)
    assert out["Bo_RB_per_STB"] == pytest.approx(1.29269, rel=REL)


def test_gas_fvf():
    # Bg = 0.02827 x 0.9 x (180 + 460) / 3000 = 0.00542784 rcf/scf
    #      -> 0.00542784 x 1000 / 5.615 = 0.966668 rb/Mscf
    out = calc.gas_fvf(z=0.9, temp_f=180, pressure_psia=3000)
    assert out["Bg_rcf_per_scf"] == pytest.approx(0.00542784, rel=REL)
    assert out["Bg_rb_per_mscf"] == pytest.approx(0.966668, rel=REL)


# --------------------------------------------------------------------------- #
# Dispatch layer
# --------------------------------------------------------------------------- #
def test_run_tool_happy_path():
    out = calc.run_tool("hydrostatic_pressure", {"mud_weight_ppg": 10, "tvd_ft": 10000})
    assert "error" not in out
    assert out["pressure_psi"] == pytest.approx(5200.0, rel=REL)


def test_run_tool_drops_none_arguments():
    # A UI form leaves optional fields empty; None must not reach the function.
    out = calc.run_tool("vogel_ipr", {"q_test": 500, "pwf_test": 2000, "pr": 3000,
                                      "pwf_target": None})
    assert "error" not in out
    assert "rate_at_pwf_target" not in out


def test_run_tool_unknown_name_returns_error_dict():
    out = calc.run_tool("no_such_calculator", {"a": 1})
    assert "error" in out
    assert "no_such_calculator" in out["error"]


@pytest.mark.parametrize("args", [
    {},                                                    # missing required args
    {"mud_weight_ppg": "ten", "tvd_ft": 10000},            # unparseable number
    {"mud_weight_ppg": 10, "tvd_ft": 100, "extra": 1},     # unexpected keyword
])
def test_run_tool_bad_arguments_return_error_dict(args):
    out = calc.run_tool("hydrostatic_pressure", args)
    assert isinstance(out, dict)
    assert "error" in out


# --------------------------------------------------------------------------- #
# Gating + spec consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query,expected", [
    ("Calculate the hydrostatic pressure at 10000 ft with 10 ppg mud", True),
    ("What is the OOIP for a 640 acre block?", True),
    ("compute the decline rate after 5 years", True),
    ("Calculate the hydrostatic pressure for me", False),      # keyword, no digit
    ("Who regulates Nigerian upstream in 2021?", False),       # digit, no keyword
    ("Explain the Petroleum Industry Act", False),             # neither
    ("", False),
])
def test_needs_tools_requires_a_digit_and_a_keyword(query, expected):
    assert calc.needs_tools(query) is expected


def test_needs_tools_handles_none():
    assert calc.needs_tools(None) is False


def test_tool_menu_lists_every_calculator():
    for name in calc.CALCULATORS:
        assert f"- {name}:" in calc.TOOL_MENU


def test_tool_specs_and_calculators_agree():
    assert set(calc.TOOL_SPECS) == set(calc.CALCULATORS)
    assert len(calc.CALCULATORS) == 12


def test_tool_spec_args_match_the_function_signatures():
    for name, spec in calc.TOOL_SPECS.items():
        fn, _desc = calc.CALCULATORS[name]
        params = inspect.signature(fn).parameters
        assert spec["label"], f"{name} has no UI label"
        for arg in spec["args"]:
            argname = arg["name"]
            assert argname in params, f"{name}: '{argname}' is not a parameter of {fn.__name__}"
            assert arg.get("label"), f"{name}.{argname} has no UI label"
            has_default = params[argname].default is not inspect.Parameter.empty
            assert bool(arg.get("opt")) == has_default, (
                f"{name}.{argname}: 'opt' flag disagrees with the function default"
            )


def test_every_required_parameter_is_exposed_in_the_spec():
    for name, spec in calc.TOOL_SPECS.items():
        fn, _desc = calc.CALCULATORS[name]
        required = {p for p, v in inspect.signature(fn).parameters.items()
                    if v.default is inspect.Parameter.empty}
        assert required <= {a["name"] for a in spec["args"]}, f"{name} hides a required argument"
