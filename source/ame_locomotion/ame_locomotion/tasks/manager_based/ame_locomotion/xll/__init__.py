import gymnasium as gym

from ame_locomotion.tasks.manager_based.ame_locomotion import agents


gym.register(
    id="AME-XLL-Stage0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage0-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage0PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMERoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stakes02-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStakes02TestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMEStage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)




gym.register(
    id="AME-XLL-Tire-Array-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMETireArrayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)


gym.register(
    id="AME-XLL-Tire-Array-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_xll:XLLAMETireArrayPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.xll_ame_rsl_rl_ppo_cfg:XLLAMEPPORunnerCfg",
    },
)
