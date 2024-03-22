Dieses Github Repo dient der Dokumentation der entwickelten und verwendeten Pythonskript im Rahmen der Bachelorarbeit "Training eines automatischen Spracherkennungssystems zur Qualitätssteigerung der resultierenden Transkriptionen am Fallbeispiel der Justiz"	

Pipeline/... enthält alle Skripte die zur Erstellung eines Datensatzes bis hin zum Training des Datensatzes benötigt wurden

BA_Datensatz_V2/... enthält den entwickelten Datensatz bzw. das geschrieben Datensatzladeskripts sowie die Metadatendateien. Der vollständige Datensatz ist auf huggingface gespeichert:
https://huggingface.co/datasets/LennyBijan/BA_Datensatz_V2

WandbSweep/... enthält die modifizierte Trainingspipeline mit Integration des "Sweeps" von WandB zur Hyperparameteroptimierung
