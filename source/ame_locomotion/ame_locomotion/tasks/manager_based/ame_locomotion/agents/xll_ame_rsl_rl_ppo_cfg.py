"""RSL-RL settings for the XLL AME rough-terrain task."""

from isaaclab.utils import configclass

from .ame_rsl_rl_ppo_cfg import G1AMEPPORunnerCfg


@configclass
class XLLAMEPPORunnerCfg(G1AMEPPORunnerCfg):
    experiment_name = "xll_ame"
