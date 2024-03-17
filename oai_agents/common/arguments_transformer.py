sweep_config = {
    "name": "eye_and_gameplay",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"values": [3e-5]}, 
        "batch_size": {
            "values": [128]
        },
        "epochs": {"values": [10000]},
        "decay_step_size": {"values": [10]},
        "decay_factor": {"values": [ 0.998]},
        "num_timesteps_to_consider": {"values": [20]},
        "agent_name": {"values": ['haha', 'random_agent', 'selfplay']},
        "layout": {"values": ['asymmetric_advantages', 'coordination_ring','counter_circuit_o_1order']}
    },
}


class TransformerConfig:
    # Transformer Model Initialization
    num_timesteps_to_consider = 20
    d_model = 512
    n_head = 8
    num_layers = 6
    dim_feedforward = 2048

    # Warmup
    warmup_steps = 2500
    base_lr = 1e-8
    max_lr = 3e-5

    decay_step_size = 10
    decay_factor = 0.9

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
