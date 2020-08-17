"# Code under testing"   

This code consists of Dynamic parallelism. To run this code modern gpu (which supports Dynamic parallelism) is required.

**compile:**
_____________
for small gpu:
```shell
nvcc -arch=sm_35 -rdc=true SSSPmainForUndirected_v1.cu -lcudadevrt -o op_main
```
for cluster(e.g. foundry) gpu:
```shell
nvcc -arch=sm_70 -rdc=true SSSPmainForUndirected_v1.cu -lcudadevrt -o op_main
```

**run:**
_____________
```shell
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name
```

example:  
```shell
./op_main graph.txt 6 7 SSSP.txt cE.txt
```


**File format:**
_____________

Original graph (3 col):
node1 node2 edge_weight

SSSP (3 col):
node parent distance

change edges file (4 col):
node1 node2 edge_weight inst_status

inst_status can be 0(for deletion) or 1(for insertion)

**How to prepare input SSSP data from original graph:**
___________________________________________________
Use below command to generate input SSSP tree:
```shell
nvcc -o op_ssspSeq seqSSSPwithDist.cpp
./op_ssspSeq original_graph_file_name number_of_nodes
```
example:  
```shell
./op_ssspSeq graph.txt 6
```

**How to prepare change edges:**
____________________________
```shell
nvcc -o op_cE createChangedEdges.cpp  
./op_cE original_graph_file_name number_of_nodes number_of_change_edges percent_of_insertion  
```
example:  
```shell
./op_cE graph.txt 6 10 50
```