import numpy as np
def activation_function(x):
    def sigmoid(element):
        return 1/(1+np.exp(-element))
    return np.vectorize(sigmoid)(x)

def generate_input_and_target(dim_input, dim_output, input_duration, num_pulses, amplitude = 1):
    #generate input
    pulse_duration = 10;
    generated_input = np.zeros([input_duration, dim_input])
    pulse_start = np.random.randint(0, input_duration, num_pulses)
    for i in range(num_pulses):
        if pulse_start[i] <= input_duration - pulse_duration:
            generated_input[pulse_start[i]:(pulse_start[i] + pulse_duration), np.random.randint(3)] = (2*(np.random.rand()<.5)-1)*amplitude
    #generate target
    target = np.zeros([input_duration, dim_output], dtype=int)
    prev = np.zeros(dim_output, dtype=int)
    for t in range(input_duration):
        changed = (generated_input[t] != 0)
        index_changed = changed.nonzero()[0]
        if index_changed.size != 0:
            prev[index_changed] = generated_input[t, index_changed]
        else:
            pass
        target[t] = prev
    return generated_input, target