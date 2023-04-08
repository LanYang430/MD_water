# TUTORIAL

## Installation

1. Install DMFF
   Follow the instructions at [DMFF](https://github.com/deepmodeling/DMFF/blob/master/docs/user_guide/installation.md) to complete the installation and testing. Once installed successfully, DMFF will be available in the `dmff` conda environment. Also, please note that this water test case was tested using the DMFF 0.2.0 version, which can be downloaded here: [DMFF 0.2.0](https://github.com/deepmodeling/DMFF/releases) 

2. Install i-PI
   Activate the "dmff" environment using the following command:

    ```bash
   conda activate dmff
    ```

   Then, refer to the "Quick Setup and Test" section at [i-PI](https://github.com/i-pi/i-pi#readme) for instructions on installing and testing i-PI.

3. Install EANN

   To install the software, please refer to the [EANN manumal_2020_7_29.pdf](https://github.com/Humourist/MD_water/blob/main/EANN manumal_2020_7_29.pdf) file located in the folder.

## MD Inputs

​       We run the MD simulation by interfacing the hybrid H2O force field with i-PI. i-PI uses a server-client system: the server is i-PI itself, which is a master process in charge of propagating dynamics. And the server gather information from client processes, which are typically energy and force calculators. The server updates the system structure, and returns the new position information to the force field calculation clients. The client listen to the server for atomic positions, then compute the forces and energy, and return it to the server. The server and the clients communicate with each other through UNIX sockets, which are a special type of file in the filesystem. 

The parameters for molecular dynamics, such as time step, simulation time scale, temperature, and output options, are specified in `input.xml`, which is the input file for the server process.

An example of `input.xml` is shown below, where the stride for output trajectory is defined:

```xml
  <output prefix='simulation'>
    <properties stride='200' filename='out'>  [ step, time{picosecond}, temperature{kelvin}, potential ] </properties>
    <trajectory filename='pos' stride='200' cell_units='angstrom'> positions{angstrom} </trajectory>
    <checkpoint stride='10'/>
  </output>
```

The input.xml file defines the forces type and their communication address (i.e., socket file names). Since the hybrid force field consists of two parts, the EANN neural network and the physical model coded in DMFF, we define two socket addresses, which the server will listen to and the client will write to. 

```xml
  <ffsocket name='dmff' mode='unix'>
    <address> unix_dmff </address>
  </ffsocket>
  <ffsocket name='EANN' mode='unix'>
    <address> unix_eann </address>
  </ffsocket>
```

`</system> `specifies the specific parameters for the simulation. 

```xml
  <system>
<initialize nbeads='32'>
      <file mode='pdb'> density_0.03338_init.pdb </file>
      <velocities mode='thermal' units='kelvin'> 295 </velocities>
    </initialize>
    <forces>
      <force forcefield='dmff' weight='1.0'> </force>
      <force forcefield='EANN' weight='1.0'> </force>
    </forces>
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <timestep units='femtosecond'> 0.50 </timestep>
        <thermostat mode='langevin'>
          <tau units='femtosecond'> 1000 </tau>
        </thermostat>
      </dynamics>
    </motion>
    <ensemble>
      <temperature units='kelvin'> 295 </temperature>
    </ensemble>
  </system>
```

`nbeads` controls the type of MD simulation, where `nbeads=1` corresponds to classical MD simulation, while `nbeads=32` corresponds to PIMD simulation.

`<forces>` defines the types of forces in the system.

`<dynamics mode='nvt'>` specifies that the simulation type is NVT (constant number of particles, volume, and temperature) simulation.

`<timestep units='femtosecond'> 0.50 </timestep>` sets the simulation time step to 0.5 fs.

You may also see a more detailed description of `input.xml` in i-PI manual



## Initiate Calculation

To initiate the calculation, submit the 'sub.sh' script using the following command

```bash
sbatch sub.sh
```

`sub.sh`will first run `run_server.sh` to start an i-PI server, which reads the MD information from `input.xml`, initializing the entire MD simulation.

```bash
# run server
bash run_server.sh &
```

Then, we initiate both the EANN and the DMFF force clients.

```bash
# run client
iclient=1
while [ $iclient -le 8 ];do
    bash run_EANN.sh &
    export CUDA_VISIBLE_DEVICES=$((iclient-1))
    bash run_client_dmff.sh &
    iclient=$((iclient+1))
    sleep 1s
done
```

In the case of PIMD with `nbeads=32`, both DMFF and EANN need to be calculated 32 times at each step. To improve simulation speed, parallel computing is usually employed by running multiple clients simultaneously. In this example, we start 8 clients, so each client compute forces 4 times in each step. For the best performance, the number of clients should be a factor of `nbeads`.

Each `client_dmff.py` computation requires a dedicated GPU. The GPU device number is specified using `export CUDA_VISIBLE_DEVICES=$((iclient-1))`, such that each client is submitted to a separate GPU.



## Environment Settings

The environment settings for running DMFF are configured in the `run_client_dmff.sh` file.

Load modules of compilers, load `cuda`for running on GPU.

    ```bash
module load gcc/8.3.0
module load fftw/3.3.8/single-threads
module load compiler/intel/ips2018/u1
module load mkl/intel/ips2018/u1
module load cuda/11.4
    ```

Sett single-threaded execution

```bash
export OMP_NUM_THREADS=1
```

The environment settings for running EANN are configured in the `run_EANN.sh` file.

Set 8 threads for concurrent execution of tasks, export `MODULEPATH` to your own path of module files. Load modules for compiling and EANN/2.0 for EANN training and testing.

```bash
export OMP_NUM_THREADS=8
export OMP_STACKSIZE=2000000
export MODULEPATH=$MODULEPATH:<your_module_path>
module load Anaconda/anaconda3/2019.10
module load compiler/intel/ips2018/u4
module load mkl/intel/ips2018/u4
module load EANN/2.0
```




## Settings in Force Calculation Clients

The computation in `client_dmff.py` involves the following steps:

1. Input of pdb file and two xml files.

   ```python
       fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
       f_xml = sys.argv[2] # xml file that defines the force field
       r_xml = sys.argv[3] # xml file that defines residues
   ```

The energy and forces are computed using `DMFFDriver`, where `admp_calculator` wraps the calculations for ELST/POL/DISP/isotropic SR together with the geometry-dependent charge/C6, allowing for fluctuated leading term in MD simulation.

```python
        def admp_calculator(positions, box, pairs):
            c0, c6_list = compute_leading_terms(positions,box) # compute fluctuated leading terms
            Q_local = params_pme["Q_local"][pme_generator.map_atomtype]
            Q_local = Q_local.at[:,0].set(c0)  # change fixed charge into fluctuated one
            pol = params_pme["pol"][pme_generator.map_atomtype]
            tholes = params_pme["tholes"][pme_generator.map_atomtype]
            c8_list = jnp.sqrt(params_disp["C8"][disp_generator.map_atomtype]*1e8)
            c10_list = jnp.sqrt(params_disp["C10"][disp_generator.map_atomtype]*1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))
            covalent_map = disp_generator.covalent_map
            a_list = (params_disp["A"][disp_generator.map_atomtype] / 2625.5)
            b_list = params_disp["B"][disp_generator.map_atomtype] * 0.0529177249

            E_pme = pme_generator.pme_force.get_energy(
                    positions, box, pairs, Q_local, pol, tholes, params_pme["mScales"], params_pme["pScales"], params_pme["dScales"]
                    )
            E_disp = disp_generator.disp_pme_force.get_energy(positions, box, pairs, c_list.T, params_disp["mScales"])
            E_sr = TT_damping_qq_disp(positions, box, pairs, params_pme["mScales"], a_list, b_list, c0, c_list[0], c_list[1], c_list[2])
            E_intra = onebodyenergy(positions, box)  # compute intramolecular energy 

            return E_pme - E_disp + E_sr + E_intra
```

 `grad` get the coordinates from i-PI and then `admp_calculator` is used to calculate the forces and energies. Finally, the calculated forces and energies are returned back to i-PI.

```python
    def grad(self, crd, cell): # receive SI input, return SI values
        positions = jnp.array(crd*1e10) # convert to angstrom
        box = jnp.array(cell*1e10)      # convert to angstrom

        # nb list
        pairs = self.nbl.update(positions)
        energy, grad = self.tot_force(positions, box, pairs)
        energy = np.float64(energy)
        grad = np.float64(grad)
        # convert to SI
        energy = energy * 1000 / 6.0221409e+23 # kj/mol to Joules
        grad = grad * 1000 / 6.0221409e+23 * 1e10 # convert kj/mol/A to joule/m
```



The short-range neural network (EANN) is calculated in `client_EANN.py`, which reads the pre-trained parameters from the `para` folder for the calculation.
