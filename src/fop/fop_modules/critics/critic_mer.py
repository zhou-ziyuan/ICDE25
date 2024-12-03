import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FOPCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FOPCritic, self).__init__()

        self.args = args
        self.n_actions = args.obs_shape
        self.n_agents = args.n_agents

        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"

        # # Set up network layers
        # self.fc1 = nn.Linear(input_shape, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 1)
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    # def forward(self, inputs):
    #     x = F.relu(self.fc1(inputs))
    #     x = F.relu(self.fc2(x))
    #     q = self.fc3(x)
    #     return q
    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            # print(inputs.size())
            # print(actions.size())
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        # state, obs, action
        #inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        # last actions
        #if self.args.obs_last_action:
        #    last_action = []
        #    last_action.append(actions[:, 0:1].squeeze(2))
        #    last_action.append(actions[:, :-1].squeeze(2))
        #    last_action = th.cat([x for x in last_action], dim = 1)
        #    inputs.append(last_action)
        #agent id
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        # print(inputs.size())
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        
        return inputs

    def _get_input_shape(self, scheme):
        # state
        #input_shape = scheme["state"]["vshape"]
        # observation
        input_shape = scheme["obs"]["vshape"]
        # actions and last actions
        #if self.args.obs_last_action:
        #    input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # agent id
        input_shape += self.n_agents #* self.n_actions
        return input_shape
