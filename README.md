"# DynamicSSSPv2" 
compile:
_____________
nvcc -o op_main CudaSSSPmain.cu

run:
_____________
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name

example: ./op_main graph.txt 6 5 SSSP.txt cE.txt


File format:
_____________

Original graph (3 col):
node1 node2 edge_weight

SSSP (3 col):
node parent distance

change edges file (4 col):
node1 node2 edge_weight inst_status

inst_status can be 0(for deletion) or 1(for insertion)