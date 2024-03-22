import wandb

# Example sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "eval/wer"},
    "parameters": {
        "lr_scheduler_type": {"values": ["linear", "cosine", "linear_schedule_with_warmup", "cosine_with_hard_restarts_schedule_with_warmup"]},
        "num_train_epochs": {"max": 8, "min": 2},
        "learning_rate": {"max": 0.0001, "min": 2.5e-06},
        "warmup_ratio": {"max": 0.4, "min": 0.01},

    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="BA_Model_V3")