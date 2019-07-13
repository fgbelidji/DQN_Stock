import os, time
import pandas as pd
import numpy as np
from dqn import DQN, DQNstock # Custom
from trading_env import load_data, take_action
from plots import plot_x, plot_rewards
# Finance stuff
import backtrader as bt
import datetime as datetime
from sklearn import metrics, preprocessing
from sklearn.externals import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# This is a test strategy
class MyStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
        ('consec_gain', 0.02),
        ('sma_per', 20),
        ('print_log', True),
        ('seq_len', 25),
        ('reward_scaler', 10.0),
        ('reward_scale_limit', 1.0),
        ('start_cash', 10000.0),
        ('skip_frame', 5),
        ('reward_gamma', 0.9),
    )
    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.p.print_log or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self, dqn):
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataclose = self.datas[0].close
        self.datavolume = self.datas[0].volume
        self.order = None
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.p.sma_per)
        self.realized_gain = 0.0
        self.position_duration = 0
        
        self.dqn = dqn
        # Indicators for the plotting show
        # bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        # bt.indicators.StochasticSlow(self.datas[0])
        # bt.indicators.MACDHisto(self.datas[0])
        # rsi = bt.indicators.RSI(self.datas[0])
        # bt.indicators.SmoothedMovingAverage(rsi, period=10)
        # bt.indicators.ATR(self.datas[0], plot=False)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.realized_gain = self.broker.getvalue()
        self.position_duration = 0
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % 
            (trade.pnl, trade.pnlcomm))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' % (
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm
                ))
                self.last_buy = len(self)
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' % (
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm
                ))
                self.last_sell = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        if self.position: #Update unrealized position counter if exists
            self.position_duration = self.position_duration + 1
        if self.order:
            return
        if len(self.dataclose) < self.p.seq_len:
            return
        elif len(self.dataclose) == self.p.seq_len:
            self.curr_state = self.get_state() #format: (1, num_rows, 5)
            # Predict+take next action
            self.action = self.dqn.act(self.curr_state)
            if self.action == 1:
                self.log('CREATE BUY, %.2f' % self.dataclose[0])
                self.order = self.buy()
            elif self.action == 2:
                self.log('CREATE SELL, %.2f' % self.dataclose[0])
                self.order = self.sell()
            else:
                return
        elif len(self.dataclose) > self.p.seq_len:
            self.new_state = self.get_state()
            self.reward = self.get_reward()
            # 1) Remember, 2) Replay, and *sometimes* 3) Target train
            self.dqn.remember(self.curr_state, self.action, self.reward, self.new_state)
            self.dqn.replay()
            if len(self.dataclose) % self.dqn.retrain_edge == 0:
                self.dqn.target_train()
            # Update step and take new action
            self.curr_state = self.new_state
            self.action = self.dqn.act(self.curr_state)
            if self.action == 1:
                self.log('CREATE BUY, %.2f' % self.dataclose[0])
                self.order = self.buy()
            elif self.action == 2:
                self.log('CREATE SELL, %.2f' % self.dataclose[0])
                self.order = self.sell()
            else:
                return
        # if not self.position:
        #     # buy on % increase two days in a row
        #     if (self.dataclose[0]/self.dataclose[-1] >= 1 - self.p.consec_gain) & (self.dataclose[-1]/self.dataclose[-2] >= 1 - self.p.consec_gain):
        #     # if self.dataclose[0] > self.sma[0]:
        #         self.log('CREATE BUY, %.2f' % self.dataclose[0])
        #         self.order = self.buy()
        # else:
        #     # if len(self) >= (self.last_buy + self.p.exitbars): # Sell 5 timesteps after last buy
        #     if self.dataclose[0] < self.sma[0]:
        #         self.log('SELL CREATE, %.2f' % self.dataclose[0])
        #         self.order = self.sell()
    
    def stop(self):
        self.log('consec_gain: %.3f, End value: %.2f' % 
            (self.p.consec_gain, self.broker.getvalue()), doprint=True)
        self.dqn.save_model('models/stock.h5')
    
    def get_state(self, firsttime=False):
        open_arr = []
        high_arr = []
        low_arr = []
        close_arr = []
        vol_arr = []
        for i in range(self.p.seq_len):
            index = i * -1
            open_arr.append(self.dataopen[index])
            high_arr.append(self.datahigh[index])
            low_arr.append(self.datalow[index])
            close_arr.append(self.dataclose[index])
            vol_arr.append(self.datavolume[index])
        stacked = np.column_stack((open_arr, high_arr, low_arr, close_arr, vol_arr))
        # Scale here
        if firsttime:
            scaler = preprocessing.StandardScaler()
            stacked = scaler.fit_transform(stacked)
            joblib.dump(scaler, 'scaler.pkl')
        else:
            scaler = joblib.load('scaler.pkl')
            stacked = scaler.fit_transform(stacked)
        stacked = np.nan_to_num(stacked)[np.newaxis, :, :] # format: (1, num_rows, num_features)
        return stacked

    def get_reward(self):
        # Realized reward
        # realized_reward = self.realized_gain/self.p.start_cash - 1
        realized_reward = self.broker.getvalue()/self.p.start_cash - 1
        # Potential-based reward
        # if self.position_duration = 0
        #     f1 = 0
        # else:
        #     # Primitive potential-based reward
        #     if self.position_duration < self.p.skip_frame:
        #         f1_p = 0 
        #         local_loop = []
        #         for i in range(self.p.skip_frame):
        #             local_loop.append()
        #         f1_f = np.average()               
        #     f1 = self.p.reward_gamma * f1_f - f1_p
        reward = realized_reward * self.p.reward_scaler
        reward = np.clip(reward, -self.p.reward_scale_limit, self.p.reward_scale_limit)
        return reward

def run_test(data_filename, seq_len, dqn):
    start_cash = 10000.0
    data = bt.feeds.YahooFinanceCSVData(
        dataname=data_filename,
        fromdate=datetime.datetime(1990, 1, 1),
        todate=datetime.datetime(2020, 12, 31),
        reverse=False)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy, seq_len=seq_len, start_cash=start_cash, dqn=dqn)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.00002)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

def main():
    global_start_time = time.time() 
    file_name = "models/rl_quant_lstm.h5"
    quote = "SPY"
    data_filename = "../data/" + quote + ".csv"
    seq_len = 30

    #Set up scalers
    load_data(data_filename, seq_len, False)

    # #Create DQN agent
    tau = 1 # % Change NNim -> NNtm
    dqn_agent = DQNstock(tau, file_name, seq_len)
    epochs = 20
    learning_progress = []

    for i in range(epochs):
        # Run new test each time
        run_test(data_filename, seq_len, dqn_agent)

    # # Load in data
    # x_data, price_data, state = load_data(data_filename, seq_len)
    # # plot_x(test_data)
    # signal = pd.Series(index=np.arange(len(x_data))) #Buy/Sell signals

    # for i in range(epochs):
    #     if i == epochs - 1: #If last epoch
    #         x_data, price_data, cur_state = load_data(data_filename, seq_len, test=True)
    #     else:
    #         x_data, price_data, cur_state = load_data(data_filename, seq_len)
    #     terminal_state = 0
    #     time_step = 1

    #     while(terminal_state == 0):
    #         #Predict action using DQN (or random)
    #         action = dqn_agent.act(cur_state)
    #         time_step += 1

    #         # Take action and update state to newer version
    #         # (Will eventually update to live-incoming data-point - i.e. for test, minute data)
    #         new_state, reward, terminal_state, signal = take_action(cur_state, action, x_data, price_data, signal, time_step)

    #         # Train, remember, and reorient goals
    #         dqn_agent.remember(cur_state, action, reward, new_state, terminal_state)
    #         dqn_agent.replay()
    #         if time_step % 20 == 0: #Re-update NN after every __ steps
    #             dqn_agent.target_train()
            
    #         cur_state = new_state

    #     # eval_reward = evaluate_Q(x_data, dqn_agent, price_data, i)
    #     eval_reward = reward
    #     learning_progress.append((eval_reward))
    #     print("Epoch #: %s, Reward: %f" % (i,eval_reward))

    # print("Completed in %f" % (time.time() - global_start_time))
    # plot_rewards(learning_progress)

if __name__ == "__main__":
    main()