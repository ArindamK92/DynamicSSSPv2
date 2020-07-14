"# DynamicSSSPv2" 
compile:
_____________
nvcc -o op_main CudaSSSPmain.cu

run:
_____________
./op_main <original graph file name> <number of nodes> <number of edges> <input SSSP file name> <change edge file name>




File format:
_____________

Original graph (3 col):
<node1> <node2> <edge weight>

SSSP (3 col):
<node> <parent> <distance>

change edges file (4 col):
<node1> <node2> <edge weight> <0(for deletion)/1(for insertion)>

