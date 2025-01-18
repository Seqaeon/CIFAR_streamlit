############################################################ Random connection (MNIST original) #################################################################

import ao_arch as ar


description = "Basic MNIST"
arch_i = [8 for i in range(28 * 28)]
arch_z = [4]
arch_c = []

connector_function = "rand_conn"
# used 360, 180 before to good success
connector_parameters = [392, 261, 784, 4]
arch = ar.Arch(
    arch_i, arch_z, arch_c, connector_function, connector_parameters, description
)









############################################################ Nearest neighbour connection (4,4) #################################################################

import ao_arch as ar


neurons_x = 28  # Number of neurons in the x direction (global variable)
neurons_y = 28  # Number of neurons in the y direction
description = "nearest_neighbour MNIST"  # Description of the agent

# Initialize the input and output architecture with 4 neurons per channel
arch_i = [8 for i in range(28 * 28)]
arch_z = [4]
arch_c = []
connector_function = "nearest_neighbour_conn"  # Function for connecting neurons


Z2I_connections = True #wether want Z to I connection or not. If not specified, by default it's False. 
# if Z and Q both have different sizes, only then define below variables
z21_random = False

connector_parameters = [4, 4, neurons_x, neurons_y, Z2I_connections, z21_random]  #ax, dg, neurons_x, neurons_y and Z2I connection (True or default False)

# Create the architecture using the Arch class from the ao_arch library
arcArch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)














############################################################ Nearest neighbour connection (7,7) #################################################################

import ao_arch as ar


neurons_x = 28  # Number of neurons in the x direction (global variable)
neurons_y = 28  # Number of neurons in the y direction
description = "nearest_neighbour MNIST"  # Description of the agent

# Initialize the input and output architecture with 4 neurons per channel
arch_i = [8 for i in range(28 * 28)]
arch_z = [4]
arch_c = []
connector_function = "nearest_neighbour_conn"  # Function for connecting neurons


Z2I_connections = True #wether want Z to I connection or not. If not specified, by default it's False. 
# if Z and Q both have different sizes, only then define below variables
z21_random = False

connector_parameters = [7, 7, neurons_x, neurons_y, Z2I_connections, z21_random]  #ax, dg, neurons_x, neurons_y and Z2I connection (True or default False)

# Create the architecture using the Arch class from the ao_arch library
arcArch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)









############################################################ Full connection #################################################################

import ao_arch as ar



# Initialize the input and output architecture with 4 neurons per channel
arch_i = [8 for i in range(28 * 28)]
arch_z = [4]
arch_c = []
connector_function = "full_conn"  # Function for connecting neurons




# Create the architecture using the Arch class from the ao_arch library
arcArch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)