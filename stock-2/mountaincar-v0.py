import gym
import os
from dqn import DQN # Custom

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'

file_name = "models/mountaincar_model_2_adv.h5"

def main():
    # Our environment
    env = gym.make("MountainCar-v0")

    trials = 200
    trial_len = 500

    updateTargetNetwork = 1000
    #Initialize our DQN agent
    dqn_agent = DQN(env=env, tau=1, file_name=file_name)
    steps = []
    max_show = -50
    # Re-run environment [trial] times
    for trial in range(trials):
        print("Trial {}".format(trial))
        # Start car at on every trial start
        cur_state = env.reset().reshape(1,2)
        # Local variables
        local_max = -50
        max_position = -0.4

        for step in range(trial_len):
            #Predict action using our DQN action function
            action, temp_max = dqn_agent.act(cur_state)
            max_show = max(temp_max, max_show)
            local_max = max(temp_max, local_max)
            env.render()

            # Make a move in env using predicted action
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1,2)

            # Adjust reward - i.e. Give more reward if max position reached!
            if cur_state[0][0] > max_position:
                max_position = cur_state[0][0]
                normalized_max = max_position + 0.6 # Reward range: 0 to 1
                reward = reward + 11 * (normalized_max ** 3) # incentivize closer to flag! Max reward of 10. n^3 reward
            # if done:
            #     reward = 20
            # elif step == 199: 
            #     reward = reward - 1
            # print("Reward: {}".format(reward))

            # Now remember, train, and reorient goals
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            if done: #remember twice because important
                dqn_agent.remember(cur_state, action, reward, new_state, done)          
            dqn_agent.replay()
            if step % 20 == 0:
                dqn_agent.target_train(False)
                # print("Retraining")

            cur_state = new_state
            if done: 
                break
        if step >= 199:
            print("Failed to complete trial, best q: {}, max-Pos: {}".format(local_max, max_position))
        else: 
            print("Completed in {} trials, best q: {}, max-Pos: {}".format(trial, local_max, max_position))
            print("_______________!!!!!!!!!!!!!!!!!_______________!!!!!!!!!!!!!!!!!")
            # Need to save model, so can reuse and get better over time
            dqn_agent.save_model(file_name)
            # break

if __name__ == "__main__":
    main()