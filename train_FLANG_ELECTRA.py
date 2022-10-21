from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "save_steps": 10000,
    "reprocess_input_data": False,
    "overwrite_output_dir": False,
    "num_train_epochs": 4,
    #"learning_rate": 1e-4,
    "warmup_steps": 100000,
    "train_batch_size": 96,
    "eval_batch_size": 96,
    "output_dir": 'data/FLANG_ELECTRA',
    "gradient_accumulation_steps": 1,
    "n_gpu": 2
}


train_file =  "/path-to-train-file"

test_file =  "/path-to-test-file"

model = LanguageModelingModel(
    "electra",
    "electra",
    args = train_args,
    train_file = train_file,
    generator_name="google/electra-base-generator",
    discriminator_name='google/electra-base-discriminator',
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
model.save_discriminator()
model.save_generator()
