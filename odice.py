import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
from util import DEFAULT_DEVICE, update_exponential_moving_average


EXP_ADV_MAX = 100.


def f_star(residual, name="Pearson_chi_square"):
    if name == "Reverse_KL":
        return torch.exp(residual - 1)
    elif name == "Pearson_chi_square":
        omega_star = torch.max(residual / 2 + 1, torch.zeros_like(residual))
        return residual * omega_star - (omega_star - 1)**2


def f_prime_inverse(residual, name='Pearson_chi_square'):
    if name == "Reverse_KL":
        return torch.exp(residual - 1)
    elif name == "Pearson_chi_square":
        return torch.max(residual, torch.zeros_like(residual))


class ODICE(nn.Module):
    def __init__(self, vf, policy, max_steps, f_name="Pearson_chi_square", Lambda=0.8, eta=1.0,
                 use_twin_v=False, value_lr=1e-4, policy_lr=1e-4, weight_decay=1e-5, discount=0.99, beta=0.005):
        super().__init__()
        self.vf = vf.to(DEFAULT_DEVICE)
        self.vf_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=weight_decay)
        self.state_feature = []
        self.Lambda = Lambda
        self.eta = eta
        self.f_name = f_name
        self.use_twin_v = use_twin_v
        self.discount = discount
        self.beta = beta
        self.step = 0

    def orthogonal_true_g_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_v = self.vf_target(observations)
            target_v_next = self.vf_target(next_observations)

        v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)
        v_next = self.vf.both(next_observations) if self.use_twin_v else self.vf(next_observations)

        forward_residual = rewards + (1. - terminals.float()) * self.discount * target_v_next - v
        backward_residual = rewards + (1. - terminals.float()) * self.discount * v_next - target_v
        forward_dual_loss = torch.mean(self.Lambda * f_star(forward_residual, self.f_name))
        backward_dual_loss = torch.mean(self.Lambda * self.eta * f_star(backward_residual, self.f_name))
        pi_residual = forward_residual.clone().detach()
        td_mean, td_min, td_max = torch.mean(forward_residual), torch.min(forward_residual), torch.max(forward_residual)

        self.v_optimizer.zero_grad(set_to_none=True)
        forward_grad_list, backward_grad_list = [], []
        forward_dual_loss.backward(retain_graph=True)
        for param in list(self.vf.parameters()):
            forward_grad_list.append(param.grad.clone().detach().reshape(-1))
        backward_dual_loss.backward()
        for i, param in enumerate(list(self.vf.parameters())):
            backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[i])
        forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
        parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad), 1e-10)).item()  # avoid zero grad caused by f*
        forward_grad = (1 - parallel_coef) * forward_grad + backward_grad

        param_idx = 0
        for i, grad in enumerate(forward_grad_list):
            forward_grad_list[i] = forward_grad[param_idx: param_idx + grad.shape[0]]
            param_idx += grad.shape[0]

        self.v_optimizer.zero_grad(set_to_none=True)
        torch.mean((1 - self.Lambda) * v).backward()
        for i, param in enumerate(list(self.vf.parameters())):
            param.grad += forward_grad_list[i].reshape(param.grad.shape)

        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.vf_target, self.vf, self.beta)

        # Update policy
        weight = f_prime_inverse(pi_residual, self.f_name)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # wandb
        if (self.step + 1) % 10000 == 0:
            wandb.log({"v_value": v.mean(), "weight_max": weight.max(), "weight_min": weight.min(),
                       "td_mean": td_mean, "td_min": td_min, "td_max": td_max, }, step=self.step)

        self.step += 1

    def true_g_update(self, observations, actions, next_observations, rewards, terminals):
        v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)
        v_next = self.vf.both(next_observations) if self.use_twin_v else self.vf(next_observations)

        residual = rewards + (1. - terminals.float()) * self.discount * v_next - v
        dual_loss = f_star(residual, self.f_name)
        pi_residual = residual.clone().detach()
        td_mean, td_min, td_max = torch.mean(residual), torch.min(residual), torch.max(residual)

        v_loss = torch.mean(((1 - self.Lambda) * v + self.Lambda * dual_loss))
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.vf_target, self.vf, self.beta)

        # Update policy
        weight = f_prime_inverse(pi_residual, self.f_name)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # wandb
        if (self.step + 1) % 10000 == 0:
            wandb.log({"v_value": v.mean(), "weight_max": weight.max(), "weight_min": weight.min(),
                       "td_mean": td_mean, "td_min": td_min, "td_max": td_max, }, step=self.step)

        self.step += 1

    def semi_g_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_v_next = self.vf_target(next_observations)

        v = self.vf.both(observations) if self.use_twin_v else self.vf(observations)

        TD_error = rewards + (1. - terminals.float()) * self.discount * target_v_next - v
        dual_loss = f_star(TD_error, self.f_name)
        pi_residual = TD_error.clone().detach()
        td_mean, td_min, td_max = torch.mean(TD_error), torch.min(TD_error), torch.max(TD_error)

        v_loss = torch.mean(((1 - self.Lambda) * v + self.Lambda * dual_loss))
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.vf_target, self.vf, self.beta)

        # Update policy
        weight = f_prime_inverse(pi_residual, self.f_name)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # wandb
        if (self.step + 1) % 10000 == 0:
            wandb.log({"v_value": v.mean(), "weight_max": weight.max(), "weight_min": weight.min(),
                       "td_mean": td_mean, "td_min": td_min, "td_max": td_max, }, step=self.step)

        self.step += 1

    def get_activation(self):
        def hook(model, input, output):
            self.state_feature.append(output.detach())
        return hook

    def save(self, model_dir):
        checkpoint = {
            'step': self.step,
            'vf': self.vf.state_dict(),
            'vf_target': self.vf_target.state_dict(),
            'policy': self.policy.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }
        torch.save(checkpoint, model_dir + f"/eta_{self.eta}_Lambda_{self.Lambda}_checkpoint_{self.step}.pth")
        print(f"***save models to {model_dir}***")

    def load(self, model_dir, step):
        checkpoint = torch.load(model_dir + f"/eta_{self.eta}_Lambda_{self.Lambda}_checkpoint_{step}.pth")
        self.step = checkpoint['step']
        self.vf.load_state_dict(checkpoint['vf'])
        self.vf_target.load_state_dict(checkpoint['vf_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        print(f"***load the model from {model_dir}***")
