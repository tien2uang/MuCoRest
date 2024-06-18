import gym
import prance
from gym import spaces
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
from StateVector import StateVector

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size)
        self.fc2 = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size, device):

        self.visited_states = []
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.device = device

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_q_values(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.cpu().numpy()



    def remember(self, state, action, reward, next_state, done, valid_actions):
        self.memory.append((state, action, reward, next_state, done, valid_actions))
        self.remember_state(state)

    def remember_state(self, state):
        if state not in self.visited_states:
            self.visited_states.append(state)

    def prioritize_state(self, all_states):
        if np.random.rand() <= self.epsilon and 1 < 1:
            # Check if each dictionary in all_states is in visited_states
            if all(all_state in self.visited_states for all_state in all_states):

                return random.choice(self.visited_states)
            else:
                selected_states = random.choice(list(set(all_states) - set(self.visited_states)))
                self.remember_state(selected_states)
                return selected_states
        else:
            highest_avg_q_value = -float('inf')
            next_state = None
            if len(self.visited_states)==0:

                if len(all_states) > 1:
                    next_state = all_states[1]  # 2nd best state base on Init_state function
                else:
                    next_state = all_states[0]  # current state base on Init_state function
            for visited_state in self.visited_states:
                visited_state= np.array([visited_state])
                q_values = self.get_q_values(visited_state)
                avg_q_value = np.mean(q_values)
                if avg_q_value > highest_avg_q_value:
                    highest_avg_q_value = avg_q_value
                    next_state = visited_state
            return next_state

    def act(self, state_vector, valid_mapping_actions, all_actions):
        if np.random.rand() <= self.epsilon and 1<1:
            return random.choice(valid_mapping_actions)
        operation = state_vector.get_state()
        embedded_state = torch.FloatTensor(state_vector.get_state_vector()).to(self.device)
        action = None
        max_q_value = -float('inf')
        with torch.no_grad():
            act_values = self.policy_net(embedded_state).cpu().numpy()
            for index, value in enumerate(act_values):
                if all_actions[index]['operation_id'] == operation['operation_id']:
                    if max_q_value < value:
                        action = all_actions[index]['action']
                        max_q_value = value

        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, valid_actions in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device) if next_state is not None else None
            target = reward
            if not done and next_state is not None:
                next_valid_actions = [act for act in range(self.action_size) if act in valid_actions]
                target = reward + self.gamma * torch.max(
                    torch.FloatTensor([self.target_net(next_state)[a] for a in next_valid_actions]).to(self.device))
            target_f = self.policy_net(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.policy_net(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, name):
        self.policy_net.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.policy_net.state_dict(), name)


class APITestEnv(gym.Env):
    def __init__(self, openapi_spec):
        super(APITestEnv, self).__init__()
        self.openapi_spec = openapi_spec
        self.operations = self.get_all_operations()
        self.first_state = self.operations[0]

        # Example state and action space definitions
        self.observation_space = spaces.Discrete(len(self.operations))
        self.action_space = spaces.Discrete(len(self.get_all_actions()))  # Maximum number of parameters
        self.mapping_actions = self.get_all_actions()

    def get_all_actions(self):
        actions = []
        index = 0
        for operation in self.operations:
            for action in self.get_valid_actions(operation):
                mapping_action = {
                    'id': index,
                    'operation_id': operation['operation_id'],
                    'action': action,
                }
                actions.append(mapping_action)
                index += 1
        return actions

    def get_current_state(self):
        return self.first_state

    def get_valid_mapping_actions(self, state):
        valid_mapping_actions = []
        current_operation = state
        for action in self.mapping_actions:
            if action['operation_id'] == current_operation['operation_id']:
                valid_mapping_actions.append(action)
        return valid_mapping_actions

    def analyze_information(self, spec):
        operations = []
        parameters_frequency = defaultdict(int)

        for path, path_data in spec['paths'].items():
            for method, operation_data in path_data.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    operation_id = operation_data['operationId']
                    operations.append({
                        'operation_id': operation_id,
                        'method': method,
                        'path': path,
                        'parameters': operation_data.get('parameters', []),
                        'responses': operation_data.get('responses', {})
                    })

                    for parameter in operation_data.get('parameters', []):
                        param_name = parameter['name']
                        parameters_frequency[param_name] += 1

                    for response_code, response_data in operation_data.get('responses', {}).items():
                        schema = response_data.get('schema', {}).get('properties', {})
                        for response_property in schema.keys():
                            if response_property in parameters_frequency:
                                parameters_frequency[response_property] += 1

        operation_parameters = {}
        for operation in operations:
            operation_id = operation['operation_id']
            operation_parameters[operation_id] = {}

            for parameter in operation['parameters']:
                param_name = parameter['name']
                operation_parameters[operation_id][param_name] = parameters_frequency[param_name]
        return operation_parameters

    def init_first_state(self):
        operation_parameters = self.analyze_information(self.openapi_spec)
        method_priority = {'post': 5}
        self.operations = sorted(self.operations, key=lambda op: (
            (sum(operation_parameters[op['operation_id']].values()) / len(
                operation_parameters[op['operation_id']])) if len(
                operation_parameters[op['operation_id']]) > 0 else 0,
            method_priority.get(op['method'], 0)), reverse=True)
        selected_operation = self.operations[0]
        return selected_operation

    def get_all_operations(self):
        operations = []
        for path, path_data in self.openapi_spec['paths'].items():
            for method, operation_data in path_data.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    operation_id = operation_data['operationId']
                    operations.append({
                        'operation_id': operation_id,
                        'method': method,
                        'path': path,
                        'parameters': operation_data.get('parameters', []),
                        'responses': operation_data.get('responses', {})
                    })

        return operations

    def reset(self):
        self.first_state = self.init_first_state()

        valid_mapping_actions = self.get_valid_mapping_actions(self.first_state)
        return self.first_state, valid_mapping_actions, self.get_all_actions()

    def get_all_states(self):
        return self.operations

    def step(self, action, agent: DQNAgent):

        state = self.first_state

        reward = self.execute_operation(state, action)

        done = False
        next_state = None if done else agent.prioritize_state(self.get_all_states())
        valid_mapping_actions = [] if done else self.get_valid_mapping_actions(next_state)
        return next_state, reward, done, {"valid_mapping_actions": valid_mapping_actions}

    def get_valid_actions(self, state):
        # Extract valid actions (parameters) for the given operation
        current_operation = state
        parameters = current_operation.get('parameters', [])
        required_parameters = [param for param in parameters if param.get('required', True)]
        optional_parameters = [param for param in parameters if param not in required_parameters]
        action = []
        for optional_parameter in optional_parameters:
            action.append(required_parameters + [optional_parameter])
        action.append(required_parameters)
        return action

    def execute_operation(self, state, action):
        # Simulate the execution of the operation with the given parameters
        # This function should be implemented to actually call the API and get the response
        # For now, let's just simulate a reward
        return np.random.rand()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # Load OpenAPI spec (this is a placeholder; you need to load your actual OpenAPI spec)
    # openapi_spec_file = sys.argv[1]
    openapi_spec_file = "spec/person.yaml"
    openapi_spec = prance.ResolvingParser(openapi_spec_file).specification

    env = APITestEnv(openapi_spec)
    state_size = 21  # The state is represented as an index of the operation
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size, action_size, device)

    episodes = 1000
    batch_size = 32
    target_update = 10  # Update the target network every 10 episodes

    for e in range(episodes):
        state, valid_mapping_actions, all_actions = env.reset()
        state_vector = StateVector(state)  # Convert state to a compatible format for the network
        for time in range(500):
            action = agent.act(state_vector, valid_mapping_actions, all_actions)
            next_state, reward, done, info = env.step(action, agent)
            next_state = StateVector(next_state) if next_state is not None else None
            valid_mapping_actions = info["valid_mapping_actions"]
            agent.remember(state, action, reward, next_state, done, valid_mapping_actions)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % target_update == 0:
            agent.update_target_network()
        if e % 50 == 0:
            agent.save(f"dqn_{e}.pth")
