from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "save_steps": 20000,
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    #"learning_rate": 1e-4,
    "warmup_steps": 10000,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "output_dir": '/datadrive/finance/electra/outputs_mask2',
    "gradient_accumulation_steps": 2,
}

data_path = "/home/azureuser/finance/data" 
train_file = data_path + "/train.txt"
test_file = data_path + "/valid.txt"

model = LanguageModelingModel(
    "electra",
    #"/datadrive/finance/electra/out/checkpoint-142000",
    "electra",
    args = train_args,
    train_file = train_file,
    generator_name="google/electra-large-generator",
    discriminator_name='google/electra-large-discriminator',
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
model.save_discriminator()
model.save_generator()
