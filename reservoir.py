import numpy as np

def show_progress(s, i, loops):
    if(i%int(loops/100)==0): print(s, "progress:", int(i/(loops/100)),"%")

def activation_function(x):
    def sigmoid(element):
        return 1/(1+np.exp(-element))
    return np.vectorize(sigmoid)(x)

class new_network:    
    def __init__(self, dim_input, dim_reservoir, dim_output, var_of_weight_recurrent = 1.5, weight={}):
        self.dim_input = dim_input
        self.dim_reservoir = dim_reservoir
        self.dim_output = dim_output
        var = var_of_weight_recurrent
        if(weight =={}):
            weight_recurrent = np.random.normal(0, np.sqrt(var**2/dim_reservoir), [dim_reservoir, dim_reservoir])
            # feedback
            weight_feedback = 2.0*(np.random.rand(dim_reservoir, dim_output) - 0.5)
            # readout
            weight_readout =  np.zeros([dim_output, dim_reservoir])
            # input weights
            weight_input = 2.0*(np.random.rand(dim_reservoir, dim_input) - 0.5)
            self.weight = {"input": weight_input, 
                           "readout": weight_readout,
                           "recurrent": weight_recurrent,
                           "feedback": weight_feedback}
        else:
            self.weight = weight
        
    def run(self, input_for_one_interval, reservoir_state):
        x = reservoir_state
        tau = 10
        w_inp = self.weight["input"]
        w_out = self.weight["readout"]
        w_rec = self.weight["recurrent"]
        w_feed = self.weight["feedback"]
        #run echo state network
        value_inp = input_for_one_interval.reshape([self.dim_input, 1])
        value_res = activation_function(x)
        value_out = w_out.dot(value_res)
        x_delta = -x + w_rec.dot(value_res) + w_feed.dot(value_out) + w_inp.dot(value_inp)
        x += x_delta/tau
        return x, value_res, value_out
    
    def train(self, inp, target):
        w_inp = self.weight["input"]
        w_out = self.weight["readout"]
        w_rec = self.weight["recurrent"]
        w_feed = self.weight["feedback"]
        input_duration = inp.shape[0]
        #initial state
        alpha = 1
        reservoir_state = 0.5*np.random.randn(self.dim_reservoir, 1)
        P = (1.0/alpha)*np.eye(self.dim_reservoir)
        #time constant
        tau = 10
        train_delta = 2
        for t in range(input_duration):
            show_progress("training", t, input_duration)
            x, value_res, value_out = self.run(inp[t], reservoir_state)
            reservoir_state = x        
            if(t%train_delta == 0):
                k = P.dot(value_res)
                rPr = value_res.T.dot(k)
                c = 1/(1 + rPr)
                P = P - k.dot(k.T)*c
                error = value_out - np.expand_dims(target[t], 1)
                w_out_delta = -error*c*k.T
                w_out += w_out_delta
        self.weight["readout"] = w_out
        return None
    
    def clear_weight_readout(self):
        self.weight["readout"] = np.zeros([self.dim_output, self.dim_reservoir])
        return None
    
    def test(self, inp, give_me_reservoir_state = False):
        duration = inp.shape[0]
        reservoir_state = np.zeros([self.dim_reservoir, 1])
        output = np.empty([duration, self.dim_output])
        reservoir_state_record = np.empty([duration, self.dim_reservoir])
        for t in range(duration):
            show_progress("testing", t, duration)
            reservoir_state, value_res, value_out = self.run(inp[t], reservoir_state)
            output[t] = value_out[:, 0]
            reservoir_state_record[t] = reservoir_state[:, 0]
        if(give_me_reservoir_state == False):
            return output
        else:
            return output, reservoir_state_record