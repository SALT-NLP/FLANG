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
    "train_batch_size": 90,
    "eval_batch_size": 80,
    "output_dir": 'data/spanbert',
    "gradient_accumulation_steps": 2,
    "n_gpu": 1
}


train_file =  "/path_to_train_file"
test_file =  "/path_to_test_file"

model = LanguageModelingModel(
    "auto",
    "bert-base-uncased",
    args = train_args,
    train_files = train_file,
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
model.save_model()
