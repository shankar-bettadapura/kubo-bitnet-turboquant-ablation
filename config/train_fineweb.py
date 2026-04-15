dataset = 'fineweb-edu'
out_dir = 'out-fineweb'

n_layer = 6
n_head = 6
n_embd = 384
block_size = 512
dropout = 0.1

batch_size = 8
gradient_accumulation_steps = 16
max_iters = 7630
lr_decay_iters = 7630
min_lr = 1e-4
learning_rate = 1e-3
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False
eval_interval = 500
eval_iters = 20
log_interval = 100