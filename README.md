# Qneural

## Ideal behind this

Due to the fast development of machine learning tech (algorithm, training method, development packages, etc.), data processing of quantum computing and information processing experiment are going thought a *rule-changed* revolution to improve the precision of quantum control and measurement from classical machinery. 

This package is a simple trail for employing simple but useful machine learning algorithm and training method to improve measurement and control in solid state quantum computing platform, e.g, gate defined quantum dots based semiconductor spin qubit and circuit QED based superconducting qubit.

We have the following expectation of this package:

1. A simple and general input protocol.
2. Easy training and test.
3. Major solving the following problems in experiment
   1. High assign fidelity for reflectometry (both SC and QD) readout. Directly working on IQ sequence from a single shot measurement
      - Multi-tone readout
   2. Q learning/deep Q network method in reinforcement learning to optimized control pulse parameter, leading to higher fidelity
   3. Residual neural network for pulse predistortion calibration.
   4. Computer vision assistant auto tune script for quantum dot.
   5. Keep thinking...

## Reference

- Entanglement engineering of optomechanical systems by reinforcement learning https://arxiv.org/pdf/2406.04550
- https://arxiv.org/pdf/2406.12195


<!-- Export to requirement.txt `conda list -e > requirements.txt` -->