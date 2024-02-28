sweep_config = {
    "name": "eye_and_gameplay",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"values": [3e-5]}, 
        "batch_size": {
            "values": [128]
        },
        "epochs": {"values": [700]},
        "decay_step_size": {"values": [50]},
        "decay_factor": {"values": [ 0.9]},
        "num_timesteps_to_consider": {"values": [64]} 
    },
}


class TransformerConfig:
    # Transformer Model Initialization
    num_timesteps_to_consider = 50
    d_model = 512
    n_head = 8
    num_layers = 8
    dim_feedforward = 2048

    # Warmup
    warmup_steps = 400
    base_lr = 1e-8
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
