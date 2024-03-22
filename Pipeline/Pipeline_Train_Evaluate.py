from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import torch
from jiwer import wer, cer
import wandb
import os

# Konfiguration für GPU-Nutzung und Speicher
PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb=256
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.cuda.empty_cache()

# Authentifizierungstokens laden
token_read = "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"
token_write = "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"

# Whisper-Modell für die Sprachgenerierung laden
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.generation_config.language = "de"  # Spracheinstellung

# Lade den Datensatz
common_voice = DatasetDict()
common_voice["train"] = load_dataset("LennyBijan/BA_Datensatz_V2", split="train", trust_remote_code=True)
common_voice["test"] = load_dataset("LennyBijan/BA_Datensatz_V2", split="test", trust_remote_code=True)

# Initialisiere Feature Extractor und Tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="German", task="transcribe")

# Prozessor initialisieren
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="German", task="transcribe")

# Audiodatenspalte in Datensatz umwandeln
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Funktion zur Datenvorbereitung
# Prepare Function
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    batch["input_length"] = len(batch["audio"]['array'])

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Vorbereitungsfunktion auf Datensatz anwenden
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

#  Creating DataCollator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Initializing DataCollator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    word_error_rate = 100 * wer(label_str, pred_str)
    char_error_rate = 100 * cer(label_str, pred_str)

    return {"wer": word_error_rate, "cer": char_error_rate}

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

os.environ["WANDB_PROJECT"]="BA_Model_V3"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"


# Trainingseinstellungen
training_args = Seq2SeqTrainingArguments(
    output_dir="BA_Model_V3/",
    hub_model_id="LennyBijan/BA_Model_V3",
    run_name="WB-FinalFinalRun",
    overwrite_output_dir='True',
    lr_scheduler_type="cosine_with_restarts",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,  
    learning_rate=1e-6,
    num_train_epochs=8,
    warmup_ratio=0.2,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    generation_max_length=225,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    fp16_full_eval=True,
    report_to="wandb",
    metric_for_best_model="wer",
    greater_is_better=False,
    auto_find_batch_size=True,
    hub_strategy="end",
    save_safetensors=True,
    load_best_model_at_end=True,
    push_to_hub=True,

)

# Trainer initialisieren und Training starten
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.push_to_hub("BA_Model_V3")
tokenizer.push_to_hub("BA_Model_V3")