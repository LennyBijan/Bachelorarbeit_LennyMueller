import datasets
import csv
import json
import os
import csv

_DATA_URL = "https://huggingface.co/datasets/LennyBijan/BA_Datensatz_V2/resolve/main/data"

class BA_Datensatz_V2(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Deutscher Datensatz, der sich auf juristische Daten konzentriert",
            features=datasets.Features({
                "clip_id": datasets.Value("string"),  # Die ID des Clips
                "path": datasets.Value("string"),  # Der Pfad zur Audiodatei
                "audio": datasets.Audio(sampling_rate=16_000),  # Die Audiodatei selbst
                "sentence": datasets.Value("string"),  # Der transkribierte Satz
                "split": datasets.Value("string")  # Der Datensatz-Split (Train/Test)
            }),
            supervised_keys=None,
        )
    # Generiert Split-Generatoren, die angeben, wie die Daten für die verschiedenen Splits geladen werden.
    def _split_generators(self, dl_manager):
        dl_manager.download_config.ignore_url_params = True

        # Initialisieren von Pfaden und Variablen für Trainings- und Testdaten
        audio_path = {}
        local_extracted_archive = {}
        metadata_path = {}
        split_type = {"train": datasets.Split.TRAIN, "test": datasets.Split.TEST}

        # Durchläuft die Splits und lädt bzw. entpackt die zugehörigen Daten
        for split in split_type:
            audio_path[split] = dl_manager.download(f"{_DATA_URL}/audio_{split}.tgz")
            local_extracted_archive[split] = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
            metadata_path[split] = dl_manager.download_and_extract(f"{_DATA_URL}/metadata_{split}.csv.gz")
        path_to_clips = "BA_Datensatz_V2"

        # Erstellt und gibt die Split-Generatoren zurück
        return [
            datasets.SplitGenerator(
                name=split_type[split],
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive[split],
                    "audio_files": dl_manager.iter_archive(audio_path[split]),
                    "metadata_path": metadata_path[split],
                    "path_to_clips": path_to_clips,
                    "split": split  # Übergibt den Split-Namen an _generate_examples
                },
            ) for split in split_type
        ]
    # Generiert die Beispiele aus den geladenen Daten basierend auf der Metadaten-CSV.
    def _generate_examples(self, audio_files, metadata_path, path_to_clips, local_extracted_archive, split):
        metadata = {}
        # Öffnen und Lesen der Metadaten-CSV-Datei
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                filename, sentence = row
                clip_id = filename.split('_')[0]
                # Dynamisches Anhängen des Split-Namens an path_to_clips
                path = os.path.join(path_to_clips, split +"/wav/"+ clip_id, filename)
                metadata[path] = {
                    "clip_id": clip_id,
                    "sentence": sentence,
                    "path": path,
                }
        id_ = 0
        # Iteration durch die Audio-Dateien und Erstellung der Datensatzeinträge
        for path, file_content in audio_files:
            if path in metadata:
                result = dict(metadata[path])
                path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                audio_data = {"path": path, "bytes": file_content.read()}
                result["audio"] = audio_data
                result["path"] = path
                yield id_, result
                id_ += 1
            else:
                print(f"Kein Metadateneintrag für {path}")
