
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import load_metric
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Constants
lang1 = "texts_lang1.txt" #txt file with sentences in input language on each line. 
lang2 = "texts_lang2.txt" #txt file with sentences; in input language translation to target language on each line.
BLEU = "bleu"
TARGET = "target lang name"
TARGET_TEXT = "summaries"
EPOCH = "epoch"
INPUT_IDS = "input_ids"
FILENAME = "TranslationDataset.csv"
GEN_LEN = "gen_len"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
MODEL_CHECKPOINT = "eslamxm/mt5-base-arabic" #the model names from huggingface
MODEL_NAME = 's1'
LABELS = "labels"
PREFIX = ""
SOURCE = "source language name"
SOURCE_TEXT = "full_texts"
SCORE = "score"
SEQ2SEQName = "translation"

model_name = 's1'
lr = 0.0002
decay = 0.227812
batch_size = 32
epochs = 6

def postprocess_text(preds: list, labels: list) -> tuple:
    """Performs post processing on the prediction text and labels"""

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[SEQ2SEQName] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[SOURCE] = sr_text
        temp_dict[TARGET] = tr_text

        data_dict[SEQ2SEQName].append(temp_dict)

    return data_dict


def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 model_checkpoint: str,
                                 tokenizer: AutoTokenizer):
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs = PREFIX + row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)

        model_inputs[SEQ2SEQName] = row

        # setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
            model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)

    return preped_data



def compute_metrics(eval_preds: tuple) -> dict:
    """computes bleu score and other performance metrics """

    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {BLEU: result[SCORE]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result[GEN_LEN] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result



# Split the dataset into source (X) and target (y)
with open(lang1, 'r', encoding='utf-8') as file1: 
    X = file1.readlines()

with open(lang2, 'r', encoding='utf-8') as file2: 
    y = file2.readlines()

# Split the data into train, test, and validation sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True, random_state=100)

# Print the sizes of each set
print(f"Training set size: {len(x_train)}")
print(f"Validation set size: {len(x_val)}")
print(f"Test set size: {len(x_test)}")



# Prepare training, validation, and test data
training_data = prep_data_for_model_fine_tuning(x_train, y_train)
validation_data = prep_data_for_model_fine_tuning(x_val, y_val)
test_data = prep_data_for_model_fine_tuning(x_test, y_test)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# Generate model-ready inputs for training, validation, and test
train_data = generate_model_ready_dataset(dataset=training_data[SEQ2SEQName],
                                          tokenizer=tokenizer,
                                          source=SOURCE,
                                          target=TARGET,
                                          model_checkpoint=MODEL_CHECKPOINT)
validation_data = generate_model_ready_dataset(dataset=validation_data[SEQ2SEQName],
                                               tokenizer=tokenizer,
                                               source=SOURCE,
                                               target=TARGET,
                                               model_checkpoint=MODEL_CHECKPOINT)
test_data = generate_model_ready_dataset(dataset=test_data[SEQ2SEQName],
                                         tokenizer=tokenizer,
                                         source=SOURCE,
                                         target=TARGET,
                                         model_checkpoint=MODEL_CHECKPOINT)

# Convert to DataFrame
train_df = pd.DataFrame.from_records(train_data)
validation_df = pd.DataFrame.from_records(validation_data)
test_df = pd.DataFrame.from_records(test_data)

# Convert DataFrames to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the pre-trained model and define training arguments
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

model_args = Seq2SeqTrainingArguments(
    f"{MODEL_NAME}-finetuned-{SOURCE}-to-{TARGET}",
    evaluation_strategy=EPOCH,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=decay,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True
)

# Create a data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Initialize the Seq2SeqTrainer for fine-tuning
trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("training model")
# Commence the model training
trainer.train()
print(" model")
# Save the fine-tuned model
trainer.save_model("s1")


# Perform translations on the test dataset
test_results = trainer.predict(test_dataset)

# Obtain and display the test BLEU score
print("Test Bleu Score: ", test_results.metrics["test_bleu"])

# Translate input sentences and generate predictions
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []
test_input = test_dataset[SEQ2SEQName]

for input_text in tqdm(test_input):
    source_sentence = input_text[SOURCE]
    encoded_source = tokenizer(source_sentence,
                               return_tensors=SOURCE,
                               padding=True,
                               truncation=True)
    encoded_source.to(device)  # Move input tensor to the same device as the model

    translated = model.generate(**encoded_source)

    predictions.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])

# Move the model back to CPU if needed
model.to("cpu")
with open("/home/saughmon/IREP_Project/Pipeline/src/fine_tuning/log.txt", 'w', encoding='utf-8') as file: 
    file.write(model_name + ' done training, ' + 'with bleu score: ' + str(test_results.metrics["test_bleu"])+ '\n')


print('doneeee')