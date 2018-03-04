import numpy as np
import numba as nb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from network_structure.reservoir import reservoir
from matplotlib import cm
from copy import deepcopy

def enlarge(array, scaling):
    array = array.reshape([-1,1])
    for i in range(len(array)):
        index = i*scaling
        array = np.insert(array, index, array[index]*np.ones(scaling-1))
    return array

def repeat(array, repeat_time, num_type=float):
    array_new = np.empty(len(array)*repeat_time, num_type)
    for i in range(repeat_time):
        index = i*len(array)
        array_new[index:index+len(array)] = array
    return array_new
    
class cluster():
    def __init__(self, size_cluster):
        self.size_cluster = size_cluster
    
    def generate_spatial_structure(self, size_container=0, shape="cuboid", distribution="random"):
        if isinstance(size_container, int) is True:
            self.size_container = self.size_cluster
        else:
            self.size_container = size_container
        dim_cluster = len(self.size_cluster)
        dim_container = len(self.size_container)
        if dim_cluster is not dim_container:
            print("The dimension of the neuron cluster and the container should be same")
            return 0
        else:
            num_nodes = 1
            for axis in range(dim_cluster):
                num_nodes *= self.size_cluster[axis]                
                
        coordinates = np.empty([num_nodes, dim_cluster])
        if shape=="cuboid":
            if distribution is "random":
                for axis in range(dim_container):
                    len_container = self.size_container[axis]
                    len_cluster = self.size_cluster[axis]
                    coordinates[:, axis] = np.random.rand(num_nodes)*len_container - len_container/2
                    
            elif distribution is "uniform":
                list_len = []
                tmp = num_nodes
                for axis in range(dim_container):
                    len_container = self.size_container[axis]
                    len_cluster = self.size_cluster[axis]
                    tmp = int(tmp/len_cluster)
                    if len_cluster == 1:
                        xx = np.array([0])
                    else:
                        xx = np.linspace(-len_container/2, len_container/2, len_cluster)
                    repeat_times = tmp
                    enlarge_times = int(num_nodes/repeat_times/len_cluster)
                    coordinates[:, axis] = repeat(enlarge(xx, enlarge_times), repeat_times)
            else:
                print("unfinished")
        self.coordinates = coordinates
    
    def embedding(self, dim_manifold, axis=[]):
        axis.sort()
        dim_container = self.coordinates.shape[1]
        if dim_manifold - dim_container != len(axis):
            print("please specify which axis to be extended")
        else:
            for i in range(len(axis)):
                self.coordinates = np.insert(self.coordinates, axis[i], 0, axis=1)
                
    def set_position_and_rotation(self, position=0, rotation=0):
        dim = self.coordinates.shape[1]
        if position==0:
            position = [0]*dim
        if rotation==0:
            rotation = [0]*dim
        if len(position)==dim and len(rotation)==dim:
            self.coordinates += position
            #rotation_matrix = np.eye(dim)
            #self.coordinates *= rotation
        else:
            print("The dimension of the position or rotation matrix is not the same with the reservoir")

def distance_along_axis(x1, x2):
    return exp(-abs(x1-x2))

def distance(p1, p2, distance_function, scaling=[1,1,1]):
    dim_p1 = len(p1)
    dim_p2 = len(p2)
    if dim_p1 != dim_p2:
        print("the dimensions of the two points are not the same")
        return
    else:
        d = 1
        for axis in range(dim_p1):
            d *= distance_function(p1[axis], p2[axis])*scaling[axis]
        return d
@nb.jit
def connect_a_and_b_with_coordinate(num_a, num_b, coordinate_a, coordinate_b, distance_function, nodes_sign, scaling_weight, shifting, scaling_axis=[1,1,1]):
    if coordinate_a.shape[0] != num_a or coordinate_b.shape[0] != num_b:
        print("coordinate matrix shape wrong")
        return
    else:
        pass
    weight = np.zeros([num_b, num_a])
    for i in range(num_a):
        for j in range(num_b):
            weight[j,i] = distance(coordinate_a[i], coordinate_b[j], distance_function, scaling_axis)*nodes_sign[i]#*(np.random.randint(2)*2-1)
    return weight/max(np.max(weight), abs(np.min(weight)))*scaling_weight+shifting

class spatial_network(reservoir):
    def __init__(self, dim_manifold=3):
        nodes_coordinates = {}
        nodes_cluster = {}
        self.nodes_dict = ["inp", "res", "out"]
        self.weight_dict = ["inp2res", "res2res", "res2out", "out2res"]
        self.dim_manifold = dim_manifold
        self.nodes_cluster = nodes_cluster
        self.nodes_coordinates = nodes_coordinates
        self.num = {}
        self.value = {}
        self.weight = {}
        self.size_cluster = {}


    def set_size_of(self, nodes_type, size_cluster):
        self.size_cluster[nodes_type] = size_cluster
        self.nodes_cluster[nodes_type] = cluster(size_cluster)
        num_nodes = 1
        for i in range(len(size_cluster)):
            num_nodes *= size_cluster[i]
        self.num[nodes_type] = num_nodes
        self.value[nodes_type] = np.zeros([num_nodes, 1])

    def initial_weight(self):
        self.weight = {}
        self.sparsity = {}
        for weight_type in self.weight_dict:
            a, b = weight_type.split("2")
            self.weight[weight_type] = np.zeros([self.num[b], self.num[a]])
            self.sparsity[weight_type] = 1

        
    def generate_spatial_structure_of(self, nodes_type, container_size, shape="cuboid", distribution="random"):
        cluster = self.nodes_cluster[nodes_type]
        cluster.generate_spatial_structure(container_size, shape, distribution)
        self.nodes_coordinates[nodes_type] = cluster.coordinates
    
    def embedding_of(self, nodes_type, axis=[]):
        cluster = self.nodes_cluster[nodes_type]
        cluster.embedding(self.dim_manifold, axis)
        self.nodes_coordinates[nodes_type] = cluster.coordinates
        
    def set_position_and_rotation_of(self, nodes_type, position=0, rotation=0):
        cluster = self.nodes_cluster[nodes_type]
        cluster.set_position_and_rotation(position, rotation)
        self.nodes_coordinates[nodes_type] = cluster.coordinates
        
    def connect_with_coordinates(self, weight_name, distance_function, scaling=1, shifting=0):
        a, b = weight_name.split("2")
        self.weight[weight_name] = connect_a_and_b_with_coordinate(self.num[a], self.num[b], 
                                                                   self.nodes_coordinates[a], 
                                                                   self.nodes_coordinates[b], distance_function, self.nodes_sign[a], scaling, shifting)
        self.sparsity[weight_name] = 1

    def get_connection_situation_of(self, nodes_type, index, print_info=False, weight_range=3):
        weight = self.weight[nodes_type+"2res"]
        weight_local = weight[weight[index,:]!=0][:,weight[:,index]!=0]
        if weight_local.shape[0] == 0:
            spectral_radius = 0
        else:
            eigenvalues = np.linalg.eigvals(weight_local)
            spectral_radius = np.max(np.vectorize(np.linalg.norm)(eigenvalues))
        if print_info is True:
            print("the number of connected nodes is {0:d}".format(weight_local.shape[0]))
            print("the current spectral radius is {0:2f}".format(spectral_radius))
            print("max value:", np.max(weight_local))
            print(weight_local[0:weight_range,0:weight_range])
        else:
            return weight[index,:]!=0, spectral_radius
    @nb.jit
    def adjust_all_spectral_radius_to_value(self, nodes_type, sr=1, loops=1):
        weight_type = nodes_type + "2res"
        weight = self.weight[weight_type]
        spectral_radius_list = []
        for loop in range(loops):
            for index in range(self.num[nodes_type]):
                mask_nonzero, spectral_radius = self.get_connection_situation_of(nodes_type, index)
                index_nonzero = np.where(mask_nonzero == True)[0]
                spectral_radius_list.append(spectral_radius)
                for x in index_nonzero:
                    for y in index_nonzero:
                        weight[x, y] *= sr/spectral_radius
        self.weight[weight_type] = weight
        self.spectral_radius_list = np.asarray(spectral_radius_list)

    def plot_coordinates(self, ax, nodes_type, nodes_size, nodes_color="orange", vmin=-1, vmax=1):
        nodes_dimension =self.nodes_coordinates[nodes_type].shape[1]
        if nodes_dimension>3:
            print("The dimension of the spacial reservoir is greater than 3")
            return 0
        elif nodes_dimension==3:
            c = self.nodes_coordinates[nodes_type]
            cs = ax.scatter(c[:,0], c[:,1], c[:,2], s=nodes_size, c=nodes_color, cmap=cm.jet, vmin=vmin, vmax=vmax)
            if nodes_type is "res":
                plt.colorbar(cs)
            else:
                pass
        elif nodes_dimension==2:
            fig = plt.figure(figsize=[12,8])
            c = self.nodes_coordinates[nodes_type]
            plt.scatter(c[:,0], c[:,1], color="blue", s=nodes_size)
        else:
            print("The dimension of the spacial reservoir is less than 2")

    @nb.jit    
    def plot_weight(self, ax, weight_type, linewidth):
        weight_for_plot = self.weight[weight_type]
        wfp = (weight_for_plot - np.min(weight_for_plot))/(np.max(weight_for_plot) - np.min(weight_for_plot))
        nodes_type1, nodes_type2 = weight_type.split("2")
        nodes1 = self.nodes_coordinates[nodes_type1]
        nodes2 = self.nodes_coordinates[nodes_type2]
        weight = self.weight[weight_type]
        nodes_dimension =nodes1.shape[1]
        if nodes_dimension>3:
            print("The dimension of the spacial reservoir is greater than 3")
            return 0
        elif nodes_dimension==3:
            for i in range(len(nodes1)):
                for j in range(len(nodes2)):
                    x = nodes1[i][0], nodes2[j][0]
                    y = nodes1[i][1], nodes2[j][1]
                    z = nodes1[i][2], nodes2[j][2]
                    if weight[j,i]==0:
                        pass
                    else:
                        if wfp[j,i]>0:
                            ax.plot(x, y, z, color="red", linewidth=wfp[j,i]*linewidth)
                        else:
                            ax.plot(x, y, z, color="blue", linewidth=-wfp[j,i]*linewidth)

        elif nodes_dimension==2:
            pass
        else:
            print("The dimension of the spacial reservoir is less than 2")

    def sort_weight(self):
        coordinates = self.nodes_coordinates["res"]
        size = self.size_cluster["res"]
        sort_by_z = coordinates.T.argsort()[2]
        coordinates = coordinates[sort_by_z]
        sorty = []
        sortx = []
        bound_z = [int(ele) for ele in np.linspace(0, self.num["res"], size[2]+1)]
        for i in range(size[2]):
            lower_z, upper_z = bound_z[i], bound_z[i+1]
            coordinatesby_z = coordinates[lower_z:upper_z]
            sort_by_y = coordinatesby_z.T.argsort()[1]
            coordinatesby_z = coordinatesby_z[sort_by_y]
            bound_y = [int(ele) for ele in np.linspace(lower_z-lower_z, upper_z-lower_z, size[1]+1)]
            for j in range(size[1]):
                lower_y, upper_y = bound_y[j], bound_y[j+1]
                coordinatesby_zy = coordinatesby_z[lower_y:upper_y] 
                sort_by_x = coordinatesby_zy.T.argsort()[0]
                for ele in sort_by_x:
                    sortx.append(ele + lower_y + lower_z)
            for ele in sort_by_y:
                sorty.append(ele + lower_z)
        self.weight["res2res"] = self.weight["res2res"][sort_by_z][sorty][sortx]
        self.weight["res2res"] = self.weight["res2res"].T[sort_by_z][sorty][sortx]

        self.nodes_coordinates["res"] = self.nodes_coordinates["res"][sort_by_z][sorty][sortx]


