from params_proto import ParamsProto

class Args(ParamsProto):

    repeat_len = 2
    include_goal_in_state = False
    pin_goal=True
    shorten_ratio = 1.0

    observation_dim = 4    
    action_dim = 2
    action_weight = 1
    batch_size = 32

    bucket = None
    clip_denoised = True
    commit = None
    config = 'config.maze2d'
    dataset = 'gymnasium-corner-env-standard'
    device = 'cuda'
    diffusion = 'models.GaussianDiffusion'
    dim_mults = (1,4,8)
    ema_decay = 0.99
    exp_name = 'diffusion/H384_T256'
    gradient_accumulate_every = 2
    horizon = 128
    short_horizon = 128
    learning_rate = 0.0002
    loader = 'datasets.SequenceDataset' # 'datasets.GoalDataset'
    logbase = 'logs'
    loss_discount = 1
    loss_type = 'l2'
    loss_weights = None
    max_path_length = 200
    max_n_episodes = 1000
    model = 'models.TemporalUnet'
    n_diffusion_steps = 128
    n_reference = 50
    n_samples = 5
    n_saves = 50
    n_steps_per_epoch = 100
    n_train_steps = 15000
    normalizer = 'LimitsNormalizer'
    predict_epsilon = False
    prefix = 'diffusion/'
    preprocess_fns = []
    renderer = 'utils.Maze2dRenderer'
    sample_freq = 5000
    save_freq = 1000
    log_freq = 10
    save_parallel = False
    snapshot_root = 'all_model'
    termination_penalty = None
    use_padding = False
    forward_sample_noise = 0.1
    join_action_state = True
    trim_buffer_mode = 'kmeans'  # 'euclidean'
    render_mode = 'random_rb' # or 'expert'
    n_condition = 1 # number of pinned states at goal and start respectively
    shuffle_length = 0
    n_test = 1
    k_cluster = 200 # number of k means clusters for replay buffer generation
    d_distance = 0.5 # distance for euclidean replay buffer generation
    visualize_mode = "state" # "action" or "both"
    data_path = None
    model_type = "CDRB"
    control = False
    nickname = None # "ogdata" or "cleandata"

    cond_start = None # starting dimension for pinning states
    cond_end = None # ending dimension for pinning states
    cond_count_front = 0 # number of states to pin at the start
    goal_pin_dim = 6 
    control_goal_size = 6 # or 25
    test_validation = False
    validation_mode = "mixed"
    save_traj = 0
    set_seed = 121212

    weight_loss_by_current_radius = 1 # int {0, 1, 2, ...}, the loss is weighted as 1/({max radius of the current diffusion step for cold diffisuion} ** {this value})