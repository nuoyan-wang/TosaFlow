# Multi-Chiplet ScaheHLS-based Compiler

### Result Files Explained:

All "output" mlir from this project are found in the folder named pass_outputs. Below is an explanation of each output file, as 

##### File 1. lenet_1_fused_conv_pool.mlir:
Basic fusion pass called upon lenet, creates new layer type, used for understanding of passes, compiler, and filestructure verification.

##### File 2. lenet_2_hls_post_chiplet_loads.mlir
Seperates lenet into different chiplets based on specified chiplet loads. Here we have chip0 with load 5, chip1 with load 3, and chip2 with load 4

##### File 3. lenet_3_hls_post_chiplet_num.mlir
Seperates lenet into different chiplets based on num_chiplets. Here we have each of 2 chiplets getting half the layers.

##### File 4. resnet_post_chiplet_loads.mlir
Seperates resnet into different chiplets based on specified chiplet loads. Here we have chip0 with load 5, chip1 with load 3, and chip2 with load 4