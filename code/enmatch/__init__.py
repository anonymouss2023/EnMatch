from gym.envs.registration import register

register(
    id='HttpEnv-v0',
    entry_point='enmatch.server.httpEnv:HttpEnv',
)

register(
    id='SimpleMatchRecEnv-v0',
    entry_point='enmatch.env:RecEnvBase',
)

register(
    id='SeqSimpleMatchRecEnv-v0',
    entry_point='enmatch.env:RecEnvBase',
)
