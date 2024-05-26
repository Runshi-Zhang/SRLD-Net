from mmengine.config import Config
from mmengine.runner import Runner

config = Config.fromfile('tryunetbi8.py')
runner = Runner.from_cfg(config)
runner.train()
#runner.val()