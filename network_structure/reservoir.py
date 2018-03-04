import numpy as np
import numba as nb

def show_progress(i, loops, s="training", error=-1):
    if error == -1:
        if(i%int(loops/100)==0): print("{} progress: {}%".format(s, int(i/(loops/100))))
        else: pass
    else:
        if(i%int(loops/100)==0): print("{0} progress: {1}%, error: {2:.5f}".format(s, int(i/(loops/100)), error))


def activation_function(x, type_str):
    def sigmoid(t):
        return 1/(1+np.exp(-t))
    def tanh(t):
    	return (np.exp(t)-np.exp(-t))/(np.exp(t)+np.exp(-t))
    def relu(t):
        return np.max([0,t])
    def no_activation(t):
        return t
    return np.vectorize(eval(type_str))(x)

def generate_vector(num, distribution_type):
    if distribution_type is "zeros":
        vector = np.zeros(num)
    elif distribution_type is "ones":
        vector = np.ones(num)
    elif distribution_type is "uniform":
        vector = np.random.rand(num)
    elif distribution_type is "normal":
        vector = np.random.randn(num)
    elif distribution_type is "binary":
        vector = np.random.randint(2, size=num)
    else:
        pass
    return vector

def distance(coordinate_a, coordinate_b, sign="random"):
    if sign is "random":
        sign = np.random.randint(2)*2-1
    elif sign is "plus":
        sign = 1
    elif sign is "minus":
        sign = -1
    else:
        print("sign wrong")
        return
    return np.linalg.norm(coordinate_a, coordinate_b)*sign

def connect_a_and_b(num_a, num_b, weight_type, scaling, shifting):
    weight = generate_vector(num_a*num_b, weight_type).reshape([num_b, num_a])
    return weight*scaling+shifting


def cut_with_threshold(ndarray, threshold_lower, threshold_upper):
    new_ndarray = np.zeros(ndarray.shape)
    new_ndarray[np.logical_and(ndarray >= threshold_lower, ndarray <= threshold_upper)] = ndarray[np.logical_and(ndarray >= threshold_lower, ndarray <= threshold_upper)] 
    new_ndarray[np.logical_and(ndarray >= -threshold_upper, ndarray <= -threshold_lower)] = ndarray[np.logical_and(ndarray >= -threshold_upper, ndarray <= -threshold_lower)]
    num_zero = np.sum(new_ndarray==0)
    num_nonzero = np.sum(new_ndarray!=0)
    sparsity = num_nonzero/(num_nonzero+num_zero)
    return new_ndarray, sparsity

class reservoir:
    def __init__(self, num_inp=10, num_res=0, num_out=0):
        #inp, res and out represent input, reservoir and output
        self.nodes_dict = ["inp", "res", "out"]
        self.weight_dict = ["inp2res", "res2res", "res2out", "out2res"]
        self.num = {}
        self.num["inp"] = num_inp
        self.num["res"] = num_res
        self.num["out"] = num_out

        self.value = {}
        for nodes_type in self.nodes_dict:
            self.value[nodes_type] = np.zeros([self.num[nodes_type], 1])

        self.weight = {}
        for weight_type in self.weight_dict:
            a, b = weight_type.split("2")
            self.weight[weight_type] = np.zeros([self.num[b], self.num[a]])
        self.sparsity = {}


    def add_sign(self, percentage):
        percentage_dict = {"inp":1, "res":percentage, "out":1}
        nodes_sign = {}
        for nodes_type in self.nodes_dict:
            nodes_sign[nodes_type] = []
            for i in range(self.num[nodes_type]):
                if np.random.randint(100) < percentage_dict[nodes_type]*100:
                    nodes_sign[nodes_type].append(1)
                else:
                    nodes_sign[nodes_type].append(-1)
            nodes_sign[nodes_type] = np.asarray(nodes_sign[nodes_type])
        self.nodes_sign = nodes_sign

    def clear_diagonal_element(self):
        for i in range(self.num["res"]):
            self.weight["res2res"][i,i] = 0

    def initial_state_res(self, res_type, scaling=1, shifting=0, state_res=0):
        if isinstance(value["res"], int):
            self.state_res = generate_vector(self.num["res"], res_type, scaling, shifting).reshape([self.num["res"], 1])
        else:
            self.state_res = state_res

    def connect(self, weight_name, weight_type="normal", scaling=1, shifting=0):
        a, b = weight_name.split("2")
        self.weight[weight_name] = connect_a_and_b(self.num[a], self.num[b], weight_type, scaling, shifting)
        self.sparsity[weight_name] = 1


    def cut_weight_with_threshold(self, weight_name, threshold_lower=0, threshold_upper=1):
        self.weight[weight_name], self.sparsity[weight_name] = cut_with_threshold(self.weight[weight_name], threshold_lower, threshold_upper)

    @nb.jit
    def run(self, inp):
        value={}
        self.update = {}
        for nodes_type in self.nodes_dict:
            value[nodes_type] = self.value[nodes_type]
        self.value["inp"] = inp.reshape([self.num["inp"], 1])
        tmp = (self.weight["res2res"].dot(self.value["res"])
            +  self.weight["out2res"].dot(self.value["out"])
            +  self.weight["inp2res"].dot(self.value["inp"]))
        self.state_res = activation_function(tmp, self.__activation_function_type)
        self.value["res"] = ((1-self.__leaking_rate)*self.value["res"] + self.__leaking_rate*self.state_res)
        self.value["out"] = self.weight["res2out"].dot(self.value["res"])
        for i in range(3):
            self.update[self.nodes_dict[i]] = self.value[self.nodes_dict[i]] - value[self.nodes_dict[i]]

    def clear_value(self):
        for nodes_type in self.nodes_dict:
            self.value[nodes_type] = np.zeros([self.num[nodes_type], 1])

    def train(self, series_inp, series_tar, training_speed=0.2):
        duration, num_inp = series_inp.shape
        I = np.eye(self.num["res"])
        error_list = []
        for t in range(duration):
            inp = series_inp[t].reshape([self.num["inp"], 1])
            tar = series_tar[t].reshape([self.num["out"], 1])
            self.run(inp)
            #FORCE training(online training)
            Iv = I.dot(self.value["res"])
            vIv = self.value["res"].T.dot(Iv)
            c = 1/(1+vIv)
            I -= Iv.dot(Iv.T)*c
            error = self.value["out"] - tar
            show_progress(t, duration, s="training", error=abs(error[0][0]))
            weight_delta = -error*c*Iv.T
            self.weight["res2out"] += weight_delta*training_speed
            error_list.append(error[0][0])
        return error_list

    def test(self, series_inp):
        duration, num_inp = series_inp.shape
        series_out = np.empty([duration, self.num["out"]])
        for t in range(duration):
            show_progress(t, duration, s="testing")
            inp = series_inp[t].reshape([self.num["inp"], 1])
            self.run(inp)
            series_out[t] = self.value["out"][:,0]
        return series_out
        
    def set_leaking_rate(self, leaking_rate=0.2):
        self.__leaking_rate = leaking_rate
    def set_activation_function(self, type_str="sigmoid"):
        self.__activation_function_type = type_str

    def show_reservoir_status(self, update_without_printout=False):
        if update_without_printout == True:
             eigenvalues = np.linalg.eigvals(self.weight["res2res"])
             self.spectral_radius = np.max(np.vectorize(np.linalg.norm)(eigenvalues))
             return 
        else:
            pass
        print("size of the reservoir:", self.num["inp"], self.num["res"], self.num["out"])
        print("activation function type:", self.__activation_function_type)
        print("leaking_rate:", self.__leaking_rate)
        for name in self.weight_dict:
            if np.sum(self.weight[name]<0) != 0:
                weight_min_minus = np.min(self.weight[name][self.weight[name]<0])
                weight_max_minus = np.max(self.weight[name][self.weight[name]<0])
            else:
                weight_min_minus = 0
                weight_max_minus = 0
            if np.sum(self.weight[name]>0) != 0:
                weight_min_plus = np.min(self.weight[name][self.weight[name]>0])
                weight_max_plus = np.max(self.weight[name][self.weight[name]>0])
            else:
                weight_min_plus = 0
                weight_max_plus = 0
            print("the range of the weight " + name + " is [{0:+.5f}, {1:+.5f}] and [{2:+.5f}, {3:+.5f}], sparsity:{4:.5f}".format(weight_min_minus,
                                                                                                                                   weight_max_minus, 
                                                                                                                                   weight_min_plus, 
                                                                                                                                   weight_max_plus, self.sparsity[name]))
        eigenvalues = np.linalg.eigvals(self.weight["res2res"])
        self.spectral_radius = np.max(np.vectorize(np.linalg.norm)(eigenvalues))
        print("the current spectral radius is {0:2f}".format(self.spectral_radius))


