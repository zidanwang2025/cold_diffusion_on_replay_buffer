import numpy as np
import torch
import pdb
import copy
import os

DTYPE = torch.float
DEVICE = 'cpu'

def cycle(dl):
    while True:
        for data in dl:
            yield data

def to_device(x, device=DEVICE):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		print(f'Unrecognized type in `to_device`: {type(x)}')
		pdb.set_trace()

def batch_to_device(batch, device='cpu'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        temp_dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        )

        self.dataloader = cycle(temp_dataloader)

        self.val_dataset = val_dataset
        temp_val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        )

        temp = iter(temp_val_dataloader)
        self.n_val_batch = sum(1 for _ in temp)

        self.val_dataloader = cycle(temp_val_dataloader)
        

        temp_dataloader_vis = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        )
        self.dataloader_vis = cycle(temp_dataloader_vis)
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps):

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                train_loss, infos = self.model.loss(*batch)
                train_loss = train_loss / self.gradient_accumulate_every
                train_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {train_loss:8.4f} | {infos_str}', flush=True)

            self.step += 1
        
        # validation
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_val_batch):
                    batch = next(self.val_dataloader)
                    batch = batch_to_device(batch)

                    temp_val_loss, _ = self.model.loss(*batch)
                    val_loss += temp_val_loss
        val_loss /= self.n_val_batch
        self.model.train()
        print(f'validation_loss {self.step}: {val_loss:8.4f}')

        return train_loss, val_loss

    def save(self, file_path=None):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, file_path)
        # print(f'[ utils/training ] Saved model to {file_path}', flush=True)


