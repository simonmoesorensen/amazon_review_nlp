from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from azureml.core.run import Run


class AzureMLLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

        self.run = Run.get_context()

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in {**{"step": step}, **metrics}.items():
            self.run.log(k, v)

    @property
    def experiment(self):
        return ""

    @property
    def name(self):
        return ""

    @property
    def version(self):
        return ""
