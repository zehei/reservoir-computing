import numpy as np
import matplotlib.pyplot as plt

#extend num x to vector [x, x, ..., x].T
def generate_target(num, dim_output, duration):
    vec = np.zeros([dim_output,1])
    vec[num] = 1
    return vec.dot(np.ones([1, duration]))
#insert 0 to the adjacent pulses(pixels) with length duration
def add_zero_between_adjacent_pixels(training_data, duration=10):
    figsize = training_data.shape
    inp_insert = np.zeros([duration, figsize[1]])
    for i in range(figsize[0]):
        training_data = np.insert(training_data, 1+i*(duration+1), values=inp_insert, axis=1)
    inp = training_data/255
    return inp
def enlarge_pulse_length(inp, magnification=3):
    inp = inp.reshape([inp.shape[0], -1])
    count = 0
    duration = inp.shape[1]
    dim = inp.shape[0]
    for i in range(duration):
        index = i+count*magnification
        inp_vec = np.expand_dims(inp[:,index], 1)
        if max(inp_vec)==0:
            pass
        else:
            inp_insert = inp_vec.dot(np.ones([1, magnification])).T
            inp = np.insert(inp, axis=1, obj=index, values=inp_insert)
            count += 1
    return inp
#the shape of the input is [dim_input, duration]
def generate_input(training_data, duration=0, magnification=0, converge_time=0, inp_value=1):
    if inp_value==0:
        inp = np.zeros([training_data.shape[0], converge_time])
    else:
        inp = add_zero_between_adjacent_pixels(training_data, duration)
        inp = enlarge_pulse_length(inp, magnification)
    return inp
def plot_input(training_data, duration, magnification, converge_time):
    figsize = training_data.shape
    inp = generate_input(training_data, duration, magnification)
    inp_zero = generate_input(training_data, converge_time, inp_value=0)
    a = []
    fig = plt.figure(figsize=[8,6])
    for i in range(figsize[0]):
        a.append(plt.subplot(figsize[0], 1, i+1))
        a[i].plot(list(inp[i])+list(inp_zero[i]))
def add_bias(inp):
    if len(inp.shape)==1:
        print("the one dimensional input should have the form [1, duration]")
        return
    else:
        [dim_inp, duration] = inp.shape
        inp_bias = np.ones([1, duration])
    return np.concatenate([inp, inp_bias], axis=0)
def generate_output(r, training_data):
    #generate test input
    inp = add_bias(generate_input(training_data, duration=0, magnification=5))
    #generate test input with value 0
    inp_zero = add_bias(generate_input(training_data, converge_time=200, inp_value=0))
    #test, obtaion output value, the shape of the input signal should be [duration, dim_input]
    outp = r.test(inp.T)
    outp_zero = r.test(inp_zero.T)
    outp = np.concatenate([outp, outp_zero], axis=0)
    return outp