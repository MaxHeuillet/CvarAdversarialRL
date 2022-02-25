import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
# this is not the author's github but the package authors on which the project is based on
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        self.embedding_size = 2  # self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, action_space.n))

        # Define critic's model
        self.critic = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1))

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], -1)
        embedding = obs

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class ACModel_adversary(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self):
        super().__init__()

        self.embedding_size = 7
        self.action_space = 4

        # Define actor's model
        self.actor = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, self.action_space))

        # Define critic's model
        self.critic = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1))

        # Initialize parameters correctly
        self.apply(init_params)

    def rescale(self, state_dist, adv_dist):
        diviseur = (state_dist * adv_dist).sum(1)
        rescaled = adv_dist.T / diviseur
        return rescaled.T

    def apply_budget_constraint(self, agent_dist, perturbations, remaining_budget):

        max_perturbations = torch.max(perturbations, dim=-1, keepdim=True).values

        over_budget_trajs = max_perturbations >= remaining_budget
        empty_budget_trajs = remaining_budget - 1e-6 < 1

        lambdas = torch.zeros_like(max_perturbations)
        lambdas[over_budget_trajs] = ((max_perturbations - remaining_budget) / ((max_perturbations - 1) + 1e-6))[
            over_budget_trajs
        ]
        lambdas[empty_budget_trajs] = 1  # lambda = 1 si le budget est de 1. => aucune perturbation

        perturbations = lambdas + (1 - lambdas) * perturbations

        new_dist = perturbations * agent_dist

        return new_dist, perturbations

    def forward(self, input_info, state_dist, budgets):

        x = input_info
        x = x.reshape(x.shape[0], -1)
        embedding = x

        x = self.actor(embedding)

        dist = Categorical(logits=F.log_softmax(x, dim=1))

        perturbations_init = self.rescale(state_dist, dist.probs)
        dist, perturbations = self.apply_budget_constraint(state_dist, perturbations_init, budgets)
        dist = Categorical(probs=dist)

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
