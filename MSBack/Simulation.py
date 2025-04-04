import threading
import copy as cp
from MSToolProtein import MSToolProtein
from Utils import timer
from openmm.app import *
from openmm import *
from openmm.unit import *


"""
import os
import time
import torch
import numpy as np
import mstool
import sys
sys.path.append("/home/cwaltmann/PycharmProjects/MSBack/MSBack")
from MSToolProtein import MSToolProteinComplex
from MSToolProtein import AAProtein
from Simulation import *
"""


class QuickSim(object):
    def __init__(self, name, device=None):

        self.name = name.split(".")[0]
        pdb = PDBFile(name)
        forcefield = ForceField('charmm36.xml', 'charmm36/tip3p-pme-b.xml')
        modeller = Modeller(pdb.topology, pdb.positions)
        print('Adding hydrogens...')
        modeller.addHydrogens(forcefield)
        print('Adding solvent...')
        top = modeller.getTopology()
        top.setUnitCellDimensions((20, 20, 20))
        chains = top.chains()
        print('Minimizing...')
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)
        integrator = VerletIntegrator(0.001 * picoseconds)
        platform = Platform.getPlatformByName('OpenCL')

        properties = {'Precision': 'double'}
        if device is not None:
            properties['DeviceIndex'] = str(device)
        self.simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        self.simulation.context.setPositions(modeller.positions)

        self.topology = top
        self.forcefield = forcefield
        self.modeller = modeller
        self.system = system

    @timer
    def run_min(self, max_iterations=100):
        initial_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
        print("E0: %.3e kJ/mol" % initial_energy)
        self.simulation.minimizeEnergy(maxIterations=max_iterations)

        final_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
        print("E1: %.3e kJ/mol" % final_energy)

        positions = self.simulation.context.getState(getPositions=True).getPositions()

        PDBFile.writeFile(self.simulation.topology, positions, open(self.name + '_em.pdb', 'w'))




lock = threading.Lock()



def design_worker(mstp_start, chain_name, model, device, seqs,prefix="./temp2/"):
        try:
            prefix = "temp2/"
            mst = cp.deepcopy(mstp_start)
            mst.u.atoms = mst.u.atoms[mst.u.atoms.chain==chain_name]
            mst.write(prefix + "CA_" + chain_name + ".pdb")
            pname =prefix + "CA_" +chain_name +  ".pdb"
            mstp = MSToolProtein(pname)
            new = mstp.design_protein_sequence(output_name=prefix + "designed_CA_" + chain_name +".pdb", chroma=model, device=device, p=5 )
            with lock:
                seqs.append(list(new.get_sequence()))
        except:
            sequence = mstp_start.get_sequence()
            sequence = sequence[:int(len(sequence)/7)]
            with lock:
                seqs.append(list(sequence))



def sim_worker(name, device_index,chain, seq, prefix="./temp2/"):
        try:
            print('Loading...')
            pdb = PDBFile(name)
            forcefield = ForceField('charmm36.xml', 'charmm36/tip3p-pme-b.xml')
            modeller = Modeller(pdb.topology, pdb.positions)
            print('Adding hydrogens...')
            modeller.addHydrogens(forcefield)
            print('Adding solvent...')
            top = modeller.getTopology()                                                 
            top.setUnitCellDimensions((20,20,20))
            chains = top.chains()
            print('Minimizing...')
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)
            integrator = VerletIntegrator(0.001*picoseconds)
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'DeviceIndex': str(device_index), 'Precision': 'double'}
            simulation = Simulation(modeller.topology, system, integrator, platform, properties)
            simulation.context.setPositions(modeller.positions)
            initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            print("E0: %.3e kJ/mol" %initial_energy)
            now = time.time()
            simulation.minimizeEnergy(maxIterations=100000)    
            then = time.time()                                          
            print("run time", then-now)
            final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value           
            print("E1: %.3e kJ/mol" %final_energy)
            print('Saving...')
            
            positions = simulation.context.getState(getPositions=True).getPositions()
            
            PDBFile.writeFile(simulation.topology, positions, open(prefix + 'output' + str(chain) + '.pdb', 'w'))
            
            with lock:
                with open(prefix + "energy.log", "a") as f:
            
                    f.write(str(i) + " %.3e kJ/mol" %final_energy + " " + ''.join(seq)    + "\n")
                
            print('Done')
                
        except OpenMMException as o:
            print("caught openmm error, likely due to a clash")
            with lock:
                with open(prefix + "energy.log", "a") as f:
                    f.write(str(chain) + " 113440 kJ/mol" + " " + ''.join(seq)    + "\n")
            print('Done')


