# Monte Carlo Neuron Simulation
The premise of the Kosik Group spike sorting algorithm is that it requires neuron action potentials to be detected on two different electrodes. However, the Kilosort algorithm requires neuron action potentials to only be detected on one singular electrode.

In order to determine which fraction of action potentials are picked up by both algorithms as opposed to only by Kilosort, Neuron objects are pseudorandomly generated on the Neuropixel_Shank object in accordance with the Kratky-Porod Wormlike Chain Model for polymers. 

In each iteration of the Monte Carlo Simulation, a Neuron object is generated by randomly drawing from probability distributions informed by the literature surrounding neurons. Each neuron is incrementally grown in steps of a resolution (preset by the module to be) one hundred times finer than the sampling rate. Then, an action potential is fired and at each segment of the neuron, the voltage received at each electrode is calculated and stored in each Electrode object. Then, analysis occurs to determine if the voltages received exceed the 5rms threshold for detection at each electrode. These detection thresholds are randomly selected from a real recording with one of the arrays. Finally, the module contains functions for plotting the simulated neuron on the array, animating the action potential, and plotting the results of the monte carlo simulation.

Example usage of the module functionality in addition to a longer explanation of the Kratky-Porod Wormlike Chain Model are contained in the Example Notebook. 

Formal Results to be included in an upcoming publication coming to theaters near you are in the Monte Carlo Neuron Simulation Notebook.
