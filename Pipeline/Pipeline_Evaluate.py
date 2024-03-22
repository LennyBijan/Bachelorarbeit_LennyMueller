from transformers import pipeline
import torch
from datasets import load_dataset, DatasetDict
import datasets
import time
from evaluate import evaluator

# Define the device for the evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Clear cache
torch.cuda.empty_cache()

start_time=time.time()

# Initialize the dataset
data = DatasetDict()
data["test"] = load_dataset("LennyBijan/BA_Datensatz_V2", revision="script", split="test", trust_remote_code=True)
task_evaluator = evaluator("automatic-speech-recognition")

# Initialize the ASR pipeline/model
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    device=device,
    torch_dtype=torch_dtype,
)

task_evaluator.PIPELINE_KWARGS.pop('truncation', None)
assert 'truncation' not in task_evaluator.PIPELINE_KWARGS 


results = task_evaluator.compute(
    model_or_pipeline=asr_pipeline,
    data=data["test"],
    metric="cer"
    )

end_time = time.time()
total_runtime = end_time - start_time

print(results)
print("Total Runtime:", total_runtime, "seconds")