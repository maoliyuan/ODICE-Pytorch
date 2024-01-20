import torch
import torch.nn as nn
import torch.nn.functional as F
from util import mlp
# All networks with name {Net}Hook are used for monitoring representation of state when forwarding
# Use self.vf.fc2.register_forward_hook(self.get_activation()) to record state representation and then calculate cosine similarity
# Please check https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html for more details

class ValueFunction(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class ValueFunctionHook(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, squeeze_output=True, use_orthogonal=False):
        super().__init__()
        self.use_layer_norm = layer_norm
        self.squeeze_output = squeeze_output
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        if use_orthogonal:
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            nn.init.orthogonal_(self.fc3.weight)
        self.activation = nn.ReLU()
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state):
        x = self.activation(self.layer_norm1(self.fc1(state))) if self.use_layer_norm else self.activation(self.fc1(state))
        x = self.activation(self.layer_norm2(self.fc2(x))) if self.use_layer_norm else self.activation(self.fc2(x))
        value = self.fc3(x).squeeze(-1) if self.squeeze_output else self.fc3(x)
        return value

class TwinV(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)
        self.v2 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def both(self, state):
        return torch.stack([self.v1(state), self.v2(state)], dim=0)

    def forward(self, state):
        return torch.min(self.both(state), dim=0)[0]

class TwinVHook(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, squeeze_output=True, use_orthogonal=False):
        super().__init__()
        self.v1 = ValueFunctionHook(state_dim, layer_norm, hidden_dim, squeeze_output, use_orthogonal)
        self.v2 = ValueFunctionHook(state_dim, layer_norm, hidden_dim, squeeze_output, use_orthogonal)

    def both(self, state):
        return torch.stack([self.v1(state), self.v2(state)], dim=0)

    def forward(self, state):
        return torch.min(self.both(state), dim=0)[0]

class Discriminator(nn.Module):
    def __init__(self, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.d = mlp(dims, layer_norm=layer_norm, squeeze_output=True, output_activation=nn.Sigmoid)

    def forward(self, state):
        return self.d(state)

class RepNet(nn.Module):
    def __init__(self, state_dim, out_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), out_dim]
        self.rep = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def forward(self, state):
        return self.rep(state)

# Auto-Encoder
class AutoEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(AutoEncoder, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        z = self.mean(z)

        u = self.decode(state, z)

        return u, z

    def decode(self, state, z):
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))