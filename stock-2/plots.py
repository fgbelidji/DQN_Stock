import matplotlib.pyplot as plt

def plot_x(data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(data[:,-1, 0], label="Close")
    ax.plot(data[:,-1, 1], label="Difference")
    ax.plot(data[:,-1, 2], label="SMA15")
    ax.plot(data[:,-1, 3], label="Close-SMA15")
    ax.plot(data[:,-1, 4], label="SMA15-SMA60")
    ax.plot(data[:,-1, 5], label="RSI")
    ax.plot(data[:,-1, 6], label="ATR")
    plt.legend()
    plt.show()

def plot_rewards(rewards):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(rewards, label="Rewards")
    plt.legend()
    plt.show()