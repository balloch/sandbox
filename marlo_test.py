import marlo
import torch
import torch.nn as nn
import torchvision
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.misc import imresize
import numpy as np
 
def scale(screen_buffer, width=None, height=None, gray=False):
    processed_buffer = screen_buffer
    if gray:
        processed_buffer = screen_buffer.astype(np.float32).mean(axis=0)
        
    if width is not None and height is not None:
        return imresize(processed_buffer, (height, width))
    return processed_buffer

def get_screen(self):
        return torch.from_numpy(scale(self.game.get_state().screen_buffer, 160, 120, True))

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, True).expand_as(out))
    return out
 
def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class a2cmodel(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(a2cmodel, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 64, 8, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
 
        self.lstm_size = 256
        self.lstm = nn.LSTMCell(9600, self.lstm_size)
 
        num_outputs = num_actions
        self.critic_linear = nn.Linear(self.lstm_size, 1)
        self.actor_linear = nn.Linear(self.lstm_size, num_outputs)
 
        self.apply(init_weights)
 
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
 
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
 
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
 
        self.train()
 
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs.unsqueeze(0)))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
 
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)



if __name__ == "__main__":
    client_pool = [('127.0.0.1', 10000)]
    join_tokens = marlo.make('MarLo-FindTheGoal-v0',
            params={'client_pool': client_pool})
    assert len(join_tokens) == 1
    join_token = join_tokens[0]
    
    writer = SummaryWriter()
    
    env = marlo.init(join_token)
    obs = env.reset()
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
     
    # Game loop
    done = False
    step = 0
    while not done:
        _action = env.action_space.sample()

        obs, reward, done, info = env.step(_action)
        # writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Reward/train', reward, step)
        print('reward: ', reward)
        print('done: ', done)
        print('info: ', info)
        step += 1
    env.close()
