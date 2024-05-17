# Setting Up Config Files

Here we detail how to prepare the config files for running experiments. Please also refer to the examples provided in this repository.

## Clients

The configuration files for clients have four main sections: ``network``, ``environment``, ``adaptation`` and ``federated``.

### Network

It collects properties of the network used in the experiments. Specifically:

``model``: defines the specific architecture used (the codebase supports ``madnet2`` only)

``checkpoint``: defines the path where the pre-trained checkpoint is stored

### Environment 

It collects properties concerning the domain over which the network will run during the experiments. Specifically:

``dataset``: defines the dataset selected for the experiments (``kitti_raw``, ``drivingstereo`` or ``dsec``)

``datapath``: defines the path where ``.tar`` archives are stored

``domain``: defines the domain over which the client will run

``proxy16``: defines if proxy labels are stored in ``16bit`` or not (in our pre-processed archives, this is true for ``dsec`` only)

``subs``: defines how many sub-sequences are sampled from the domain (`-1` means all sub-sequences from the domain are sampled)

### Adaptation

It collects properties concerning the adaptation process. Specifically:

``optimizer``: defines the optimizer used

``lr``: defines the learning rate

``adapt_mode``: defines the adaptation mode (``none`` means no adaptation is carried out)

``sample_mode``: defines the sampling strategy for ``MAD``/``MAD++`` adaptation modes

``gpu``: defines the GPU id over which the client will run

### Federated

It defines the role of the client in federated experiments. Specifically: 

``sender``: if set to ``True``, the client will send its weights to the server

``listener``: if set to ``True``, the client will receive updates from the server


## Server

The configuration files for the server have three main sections: ``network``, ``adaptation`` and ``federated``.

### Network 

It collects properties of the network used in the experiments. Specifically:

``model``: defines the specific architecture used (the codebase supports ``madnet2`` only)

``checkpoint``: defines the path where the pre-trained checkpoint is stored

### Adaptation

It collects properties concerning the how the server deals with adaptation. Specifically:

``gpu``: defines the GPU id over which the server will aggregate the updates

### Federated

It collects properties concerning the federated adaptation process.
Specifically: 

``mode``: defines the data exchance policy (``fedfull`` or ``fedmad``)

``interval``: defines how often the clients send updates to the server

``bootstrap``: defines the number of updates from the server after which the listening client will start running