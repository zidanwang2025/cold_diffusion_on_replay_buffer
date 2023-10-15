from params_proto import ParamsProto

class Args(ParamsProto):

    # len for pos is 2, if you want to repeat pos and vol, change this to 4
    repeat_len = 2
    include_goal_in_state = False# BEFORE: False
    pin_goal=True
    shorten_ratio = 1.0

    state_dim = 4
    action_dim = 2
    action_weight = 1
    batch_size = 32 # BEFORE: 32

    bucket = None
    clip_denoised = True # CHANGED False
    commit = None
    config = 'config.maze2d'
    dataset = 'gymnasium-corner-env-archive'
    device = 'cpu'
    diffusion = 'models.GaussianDiffusion'
    dim_mults = (1,4,8)
    ema_decay = 0.99 # BEFORE 0.995
    exp_name = 'diffusion/H384_T256'
    gradient_accumulate_every = 2
    horizon = 128 # must be less than max_path_length
    short_horizon = 128 # must be less than max_path_length
    learning_rate = 0.0002
    loader = 'datasets.SequenceDataset'#'datasets.GoalDataset'
    logbase = 'logs'
    loss_discount = 1
    loss_type = 'l2'
    loss_weights = None
    max_path_length = 200 #40000
    max_n_episodes = 1000
    model = 'models.TemporalUnet'
    n_diffusion_steps = 128
    n_reference = 50
    n_samples = 5
    n_saves = 50
    n_steps_per_epoch = 100#10000
    n_train_steps = 15000 #1000000.0
    normalizer = 'LimitsNormalizer'
    observation_dim = 4
    predict_epsilon = False
    prefix = 'diffusion/'
    preprocess_fns = [] # originally ['maze2d_set_terminals'] but throws an error
    renderer = 'utils.Maze2dRenderer'
    sample_freq = 5000
    save_freq = 1000 #100000
    log_freq = 10
    save_parallel = False
    snapshot_root = 'all_model'#'wall_models_epsilon_False'
    # snapshot_root = 'one_goal_fixed_start'
    termination_penalty = None
    use_padding = False
    forward_sample_noise = 0.1
    render_mode = 'expert' # or 'random_rb'
    n_condition = 1 # number of pinned states at goal and start respectively
    n_test = 1
    visualize_mode = "state" # OR "action" OR "both"
    num_tests = 1000 # number of tests for action success rate
    condition = 'from_val_tail' # 'from env' or 'from val_head' or 'from_val_tail'
    prediction = 'random_rb' # or 'reconstruct': reconstruct is avaiable when condition is from val
    controller = True
    num_action_per_step = 5 # number of controller action per time step