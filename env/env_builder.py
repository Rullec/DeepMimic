from env.deepmimic_env import DeepMimicEnv
from env.deepmimic_shapevar_env import DeepMimicShapeVarEnv


class EnvType:
    NORMAL_ENV = 0
    SHAPEVAR_ENV = 1


def build_env(env_type, args, enable_draw):
    if env_type == EnvType.NORMAL_ENV:
        return DeepMimicEnv(args, enable_draw)
    elif env_type == EnvType.SHAPEVAR_ENV:
        return DeepMimicShapeVarEnv(args, enable_draw)
    else:
        print("[Error] Please Set Correct Env Type")
        exit(-1)
