import collections
from scipy.signal import butter
import numpy as np
import matplotlib.pyplot as plt

class FilterButter():
    def __init__(self, order, low_cut, high_cut, fs):
        nyquist = 0.5 * fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        if low_cut==0:
            b, a = butter(order, [high], btype='low')
            self.hist_len = order
            #len=order+1
        else:
            b, a = butter(order, [low, high], btype='band')
            self.hist_len = 2*order
            #len=order*2+1

        self.b = b
        self.a = a
        self.yhist = collections.deque(maxlen=self.hist_len)
        self.xhist = collections.deque(maxlen=self.hist_len)
        self.reset()

    def reset(self):
        self.yhist.clear()
        self.xhist.clear()
        for _ in range(self.hist_len):
            self.yhist.appendleft(0)
            self.xhist.appendleft(0)

    def filter(self, x):
        """Returns filtered x."""
        x=np.array(x)
        xs = np.array(self.xhist)
        ys = np.array(self.yhist)
        y = np.multiply(x , self.b[0]) + np.sum(
            np.multiply(xs, self.b[1:]), axis=-1) - np.sum(
            np.multiply(ys, self.a[1:]), axis=-1)
        self.xhist.appendleft(x.copy())
        self.yhist.appendleft(y.copy())
        return y

def generate_square_wave(frequency, duty_cycle, duration, amplitude, sampling_rate):
    #生成测试信号：方波
    period=1/frequency
    num_cycles = int(duration/period)
    on_time = duty_cycle * period
    #off_time = (1 - duty_cycle) * period
    
    t = np.linspace(0, duration, num=int(duration*sampling_rate), endpoint=False)
    square_wave = np.zeros_like(t)
    for i in range(num_cycles):
        square_wave[int(i*period*sampling_rate) : int((i*period*sampling_rate) + (on_time*sampling_rate))] = amplitude
    
    return t, square_wave

if __name__=="__main__":
    # Set the parameters for the square wave
    frequency = 10         # Frequency of the square wave in Hz
    duty_cycle = 0.2      # Duty cycle of the square wave (0.5 for 50% duty cycle)
    total_duration = 500/1000    # Total duration of the signal in seconds
    amplitude = 1         # Amplitude of the square wave
    sampling_rate=500

    # Generate the square wave signal
    t, square_wave = generate_square_wave(frequency, duty_cycle, total_duration, amplitude, sampling_rate)

    # Plot the square wave
    plt.plot(t, square_wave)
    plt.title('Square Wave Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('./1.jpg')


    FILTER_ORDER = 5
    FILTER_LOW_CUT = 0
    FILTER_HIGH_CUT = 10.0
    buttwoth_filter1=FilterButter(order=FILTER_ORDER,
                                  low_cut=FILTER_LOW_CUT,
                                  high_cut=FILTER_HIGH_CUT,
                                  fs=sampling_rate)
    # buttwoth_filter2=FilterButter(FILTER_ORDER,FILTER_LOW_CUT,FILTER_HIGH_CUT,sampling_rate)

    y_sig1=np.zeros_like(t)
    for index in range(len(square_wave)):
        y=buttwoth_filter1.filter(square_wave[index])
        y_sig1[index]=y

    plt.plot(t, y_sig1)
    plt.title('filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('./2.jpg')


