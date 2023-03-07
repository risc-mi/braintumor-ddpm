""" config used to train on brats """


def brats_128x128_config():
    config = {
        "cache_dir": None,
        "data": None,
        "batch_size": 32,
        "micro_batch": -1,
        "lr": 2e-04,
        "ema_rate": 0.995,
        "log_interval": 100,
        "save_interval": 10000,
        "total_steps": 400000,
        "weight_decay": 0,
        "resume_train_checkpoint": False,
        "image_size": 128,
        "in_channels": 4,
        "class_condition": False,
        "learn_sigmas": False,
        "model_channels": 64,
        "num_resnet_blocks": 3,
        "channel_mult": "1, 1, 2, 2, 4, 4",
        "attention_heads": 4,
        "attention_resolutions": "32",
        "dropout": 0,
        "diffusion_steps": 4000,
        "noise_schedule": "cosine",
        "timestep_respacing": "",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
        "sampler": "uniform"
    }
    return config

