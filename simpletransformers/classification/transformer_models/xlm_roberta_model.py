from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
)

from simpletransformers.classification.transformer_models.roberta_model import (
    RobertaForSequenceClassification,
)


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
