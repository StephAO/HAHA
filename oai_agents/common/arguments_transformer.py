sweep_config = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"distribution": "uniform", "min": 0.000001, "max": 0.00001},
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
            "min": 32,
            "q": 8,
        },
        "epochs": {"values": [5, 10, 15]},
        "decay_step_size": {"values": [30, 50, 70]},
        "decay_factor": {"values": [0.5, 0.6, 0.7, 0.8, 0.9]}

    },
}


class TransformerConfig:
    # Transformer Model Initialization
    num_timesteps_to_consider = 20
    d_model = 512
    n_head = 8
    num_layers = 8
    dim_feedforward = 2048
    num_classes = 4
    input_dim = 1260  # Based on input dimension

    # Warmup
    warmup_steps = 300
    base_lr = 1e-6
    max_lr = 1e-5

    decay_step_size = 50
    decay_factor = 0.5

    # Training Loop
    num_epochs = 10


class LSTMConfig:
    # LSTM Model Initialization
    # Model Initialization
    num_timesteps_to_consider = 20
    d_model = 512
    n_head = 8
    num_layers = 8
    dim_feedforward = 2048
    num_classes = 4
    input_dim = 1260  # Based on input dimension
    hidden_dim = 512  # Hidden dimension size
    output_dim = 4  # Number of output classes
    dropout_rate = 0.5  # Dropout rate

    # warmup

    warmup_steps = 300  # Define the number of steps for warmup
    base_lr = 1e-6  # Starting learning rate
    max_lr = 1e-5  # Target learning rate (same as the optimizer's initial lr)
    num_epochs = 300
