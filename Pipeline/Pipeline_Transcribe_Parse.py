import os
import torch
import csv
from transformers import pipeline
import whisper

# Basisverzeichnis, das Unterordner mit Audiodateien enthält, definieren
base_directory = "AudioGenFinal/train/wav/"
output_directory = "BA_Datensatz_V2/data/"

# Sicherstellen, dass das Zielverzeichnis existiert
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Pfad zur CSV-Datei definieren
csv_file_path = os.path.join(output_directory, "metadata-train.csv")

# Pipeline für automatische Spracherkennung initialisieren
# Verwendet das Whisper-Modell für Deutsch auf der verfügbaren Hardware (CUDA oder CPU)
pipe = pipeline(
    "automatic-speech-recognition",
    model="primeline/whisper-large-v3-german",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# Festlegen der Batch-Größe für die Verarbeitung
batch_size = 32

# Alle Unterordner erfassen und sortieren
all_subdirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
sorted_dirs = sorted(all_subdirs, key=lambda x: int(x.rstrip('/').split('Audio')[-1]))

# CSV-Datei zum Schreiben der Ergebnisse öffnen
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # Iteration über sortierte Unterordner
    for subdir_path in [os.path.join(base_directory, d) for d in sorted_dirs]:
        # Dateien im Unterordner sortieren
        sorted_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.wav')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))

        batch_audio_paths = []
        batch_filenames = []

        # Verarbeitung jeder Datei im sortierten Unterordner
        for filename in sorted_files:
            if len(batch_audio_paths) < batch_size:
                # Hinzufügen der Datei zum Batch
                audio_path = os.path.join(subdir_path, filename)
                batch_audio_paths.append(audio_path)
                batch_filenames.append(filename)

                # Verarbeitung des Batches, wenn er voll ist
                if len(batch_audio_paths) == batch_size:
                    # Transkription der Audiodateien im Batch
                    audios = [whisper.load_audio(path) for path in batch_audio_paths]
                    results = pipe(audios, batch_size=len(audios))

                    # Transkriptionsergebnisse in die CSV-Datei schreiben
                    for i, result in enumerate(results):
                        writer.writerow([batch_filenames[i], result["text"]])
                        print(f"Erkannter Text für {batch_filenames[i]} in CSV gespeichert.")
                    
                    # Zurücksetzen des Batches
                    batch_audio_paths = []
                    batch_filenames = []

        # Verarbeitung des letzten Batches, falls nicht leer
        if batch_audio_paths:
            # Transkription der verbleibenden Audiodateien
            audios = [whisper.load_audio(path) for path in batch_audio_paths]
            results = pipe(audios, batch_size=len(audios))

            # Schreiben der letzten Transkriptionsergebnisse in die CSV-Datei
            for i, result in enumerate(results):
                writer.writerow([batch_filenames[i], result["text"]])
                print(f"Erkannter Text für {batch_filenames[i]} in CSV gespeichert.")
