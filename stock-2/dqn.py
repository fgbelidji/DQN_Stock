import numpy as np
import random

# Neural Net stuff
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, merge
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model, load_model
from keras import backend as K
# from keras.utils import plot_model

from collections import deque

class DQN:
    # Initialize some params
    def __init__(self, env, tau, file_name):
        self.env = env
        self.memory = deque(maxlen=3000)
        
        self.gamma = 0.98 #Future rewards depreciation factor
        self.epsilon = .1 #Exporation vs eploitation (initial)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005 #standard learning rate param (0.01)
        self.tau = tau # % of NN which gets transferred to target_NN on each turn

        # NN stuff
        self.hidden1 = 24
        self.hidden2 = 48
        self.hidden3 = 24
        self.batch_size = 32

        # Initialize new model
        # self.model = self.create_adv_model() #For predicting actions
        # # "hack" implemented by DeepMind to improve convergence
        # self.target_model = self.create_adv_model() #For tracking "target values"

        # Load existing model
        # mountaincar_model_2 is {gamma:.98, lr:.005, h:24/48/24, reward: reward + 11 * (normalized_max ** 3)}
        # mountaincar_model_3 is {gamma:.95, lr:.005, h:24/48/24, reward: reward + 4 * (normalized_max ** 3), 20 if done}
        # mountaincar_model_4 is {gamma:.95, lr:.005, h:24/0/24, reward: reward + 11 * (normalized_max ** 3), 80 if done}
        self.model = load_model(file_name)
        self.target_model = load_model(file_name)

    # Simple sequential model
    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(self.hidden1, input_dim=state_shape[0], 
            activation="relu"))
        model.add(Dense(self.hidden2, activation="relu"))
        model.add(Dense(self.hidden3, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    # Create model with state value and action advantage components
    def create_adv_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        input_layer = Input(shape=(state_shape[0],))
        # Advatage NN ([action size] outputs)
        hidden1_A = Dense(self.hidden1, activation="relu")(input_layer)
        if self.hidden2 > 0:
            hidden2_A = Dense(self.hidden2, activation="relu")(hidden1_A)
        else:
            hidden2_A = hidden1_A
        hidden3_A = Dense(self.hidden3, activation="relu")(hidden2_A)
        advantage = Dense(self.env.action_space.n)(hidden3_A)
        # Value NN (1 output)
        hidden1_V = Dense(self.hidden1, activation="relu")(input_layer)
        if self.hidden2 > 0:
            hidden2_V = Dense(self.hidden2, activation="relu")(hidden1_V)
        else:
            hidden2_V = hidden1_V
        hidden3_V = Dense(self.hidden3, activation="relu")(hidden2_V)
        value = Dense(1)(hidden3_V)
        # Policy: Q(s,a) = V(s) + A(s, a) - avg(A(s, a)) -> ([action size] outputs)
        policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.env.action_space.n,))
        # subtracted = Subtract()([advantage, K.mean(advantage)])
        # policy = Add()([subtracted, value])
        model = Model(input=[input_layer], output=[policy])
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    # Take action
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        # Eplore vs exploit
        if np.random.random() < self.epsilon:
            #Take random action (expore) -> do this more in the beginning
            prediction = self.env.action_space.sample()
            # print("Random action: {}".format(prediction))
            return prediction, -50 # OpenAI-Gym random action
        else:
            #Take best possible action (expoit) -> do this more in the end
            q_table = self.model.predict(state)[0]
            prediction = np.argmax(q_table)
            # print("Action: {}".format(prediction))
            return prediction, max(q_table) #take best possible Q from predicted Q-table

    # Training: 1) remembering, 2) learning, and 3) reorienting goals

    # 1) Remember
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    # 2) Learn
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size) #take [batch_size] random elements from memory
        # Retrain immediate model on the chosen elements in memory.
        for sample in samples:
            state, action, reward, new_state, done = sample
            # Get q-table from NN
            target = self.model.predict(state) #[[q1, q2, q3]]
            if done: 
                target[0][action] = reward # This is the goal!
            else:
                # Take highest future reward (using next state)
                # Q_future = max(
                #     self.target_model.predict(new_state)[0] # returns table of Q's, max function takes highest one
                # )
                future_action = np.argmax(self.model.predict(new_state)[0])
                Q_future = self.target_model.predict(new_state)[0][future_action]

                # Use bellman equation to get better Q and replace at the action (which is index) in our table of Q's 
                target[0][action] = reward + Q_future * self.gamma
            #Retrain immediate model with current state and updated target (makes NN predict Q tables based on next state)
            self.model.fit(state, target, epochs=1, verbose=0)


    # 3) Reorient goals
    def target_train(self, done):
        # If target reached successfully, transfer much more evaluate model into target_model (assuming small tau)
        if done:
            weight_transfer = (1 - self.tau)
        else:
            weight_transfer = self.tau
        
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        # Update our target model (long-term) with weights of immediate model, which was just trained a bunch of times in replay function ^
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * weight_transfer + target_weights[i] * (1 - weight_transfer)
        self.target_model.set_weights(target_weights)

    # Save model function
    def save_model(self, name):
        self.model.save(name)

class DQNstock:
    def __init__(self, tau, file_name, seq_len):
        self.memory = deque(maxlen=300)
        
        self.gamma = 0.95 #Future rewards depreciation factor
        self.epsilon = 1 #Exporation vs eploitation (initial)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005 #standard learning rate param (0.01)
        self.tau = tau # % of NN which gets transferred to target_NN on each turn

        # NN stuff
        self.num_features = 5
        self.hidden1 = 64
        self.hidden2 = 64
        self.hidden3 = 64
        self.batch_size = 5
        self.retrain_edge = 25
        self.seq_len = seq_len

        self.num_actions = 3

        # Initialize new model
        self.model = self.create_model() #For predicting actions
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self.create_model() #For tracking "target values"

        # Load existing model
        # self.model = load_model(file_name)
        # self.target_model = load_model(file_name)

    # Build LSTM Model
    def create_model(self):
        model = Sequential()
        model.add(LSTM(
            self.hidden1,
            input_shape=(self.seq_len, self.num_features),
            return_sequences=True
        ))
        model.add(Dropout(0.25))
        model.add(LSTM(
            self.hidden2,
            # input_shape=(self.seq_len, self.num_features),
            return_sequences=False
        ))
        model.add(Dropout(0.25))
        model.add(Dense(
            output_dim=self.num_actions,
            init='lecun_uniform'
        ))
        model.add(Activation('linear'))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate)
        )

        return model

    # Take action
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        # Eplore vs exploit
        if np.random.random() < self.epsilon:
            #Take random action (expore) -> do this more in the beginning
            prediction = np.random.randint(0, self.num_actions)
            # print("Random action: {}".format(prediction))
            return prediction
        else:
            #Take best possible action (expoit) -> do this more in the end
            q_table = self.model.predict(state)[0]
            prediction = np.argmax(q_table)
            # print("Action: {}".format(prediction))
            return prediction

    # 1) Remember
    def remember(self, state, action, reward, new_state, done=False):
        self.memory.append([state, action, reward, new_state, done])

    # 2) Learn
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size) #take [batch_size] random elements from memory
        # Retrain immediate model on the chosen elements in memory.
        for sample in samples:
            state, action, reward, new_state, done = sample
            # Get q-table from NN
            target = self.model.predict(state) #[[q1, q2, q3]]
            if done: 
                target[0][action] = reward # This is the goal!
            else:
                # Take highest future reward (using next state)
                future_action = np.argmax(self.model.predict(new_state)[0])
                Q_future = self.target_model.predict(new_state)[0][future_action]

                # Use bellman equation to get better Q and replace at the action (which is index) in our table of Q's 
                target[0][action] = reward + Q_future * self.gamma
            #Retrain immediate model with current state and updated target (makes NN predict Q tables based on next state)
            self.model.fit(state, target, epochs=1, verbose=0)

    # 3) Reorient goals
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        # Update our target model (long-term) with weights of immediate model, which was just trained a bunch of times in replay function ^
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    # Save model function
    def save_model(self, name):
        self.model.save(name)

    # Plot the model
    def plot_my_dqn(self):
        # plot_model(self.model, to_file='model-imgs/model.png')
        self.model.summary()