from matplotlib import pyplot as plt
import numpy as np

def add_bias(inp, amplitude):
    if len(inp.shape)==1:
        print("the one dimensional input should have the form [1, duration]")
        return
    else:
        [dim_inp, duration] = inp.shape
        inp_bias = np.ones([1, duration])*amplitude
    return np.concatenate([inp, inp_bias], axis=0)

def show_progress(i, loops, s="training"):
    if(i%int(loops/100)==0): print("{} progress: {}%".format(s, int(i/(loops/100))))

def plot(inp, output, target, axis, start=0, end=1):
    interval = inp.shape[0]
    slide = range(int(start*interval), int(end*interval), 1)
    x = np.linspace(0, interval, interval)
    plt.plot(x[slide], inp[slide, axis], color="orange")
    plt.plot(x[slide], output[slide, axis], color="green")
    plt.plot(x[slide], target[slide, axis])

def get_value_follow_key(name, key):
	if key not in name:
		return 1
	else:
		value = 0
		_, key =  name.split(key)
		if key[0] == "0":
			pass
		elif key[1] < "9" and key[1] >= "0":
			value = int(key[0])*10 + int(key[1])
			if key[2] < "9" and key[2] >= "0":
				value = value*10 + int(key[2])
		return value/100