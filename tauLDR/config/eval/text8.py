import ml_collections

def get_config():

    pianoroll_dataset_path = 'datasets/text8'
    model_location = 'path/to/text8/checkpoints/ckpt_0000999999.pt'
    model_config_location = 'path/to/text8/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'text8'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        [['data', 'path'], pianoroll_dataset_path + '/train.npy'],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location
    config.pianoroll_dataset_path = pianoroll_dataset_path

    config.device = 'cpu'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'Text8'
    data.path = pianoroll_dataset_path + '/train.npy'
    data.S = 27
    data.batch_size = 64 #128
    data.shuffle = True
    data.shape = [20]


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping' # ConditionalTauLeaping or ConditionalPCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'uniform'
    sampler.test_dataset = pianoroll_dataset_path + '/test.npy'
    sampler.condition_dim = 32
    sampler.num_corrector_steps = 2
    sampler.corrector_step_size_multiplier = 0.1
    sampler.corrector_entry_time = 0.9
    sampler.reject_multiple_jumps = True

    return config