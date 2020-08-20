#ifndef GPUFUNCTIONS_UNDIR_V1_CUH
#define GPUFUNCTIONS_UNDIR_V1_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "all_structure_undir.cuh"
using namespace std;

#define THREADS_PER_BLOCK 1024 //we can change it

__global__ void initializeEdgedone(int* Edgedone, int totalChange) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange)
	{
		Edgedone[index] = -1;
	}
}

__global__ void deleteEdge(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, int inf, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange) {
		////Deletion case
		//for Deletion inst should be 0
		if (allChange_device[index].inst == 0) {
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			int edge_weight = allChange_device[index].edge_wt;
			Edgedone[index] = 3;
			//bool iskeyedge = false;
			//this will check if node1 is parent of node2
			//Mark edge as deleted by making edgewt = inf
			if (SSSP[node_2].Parent == node_1) {
				SSSP[node_2].Dist = inf;
				SSSP[node_2].EDGwt = inf;
				SSSP[node_2].Update = true;
			}
			else if (SSSP[node_1].Parent == node_2) {
				SSSP[node_1].Dist = inf;
				SSSP[node_1].EDGwt = inf;
				SSSP[node_1].Update = true;
			}

			//mark the edge as deleted in Adjlist
			for (int j = AdjListTracker_device[node_2]; j < AdjListTracker_device[node_2 + 1]; j++) {
				if (AdjListFull_device[j].col == node_1 && AdjListFull_device[j].wt == edge_weight) {
					AdjListFull_device[j].wt = -1;
					//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
				}

			}
			for (int j = AdjListTracker_device[node_1]; j < AdjListTracker_device[node_1 + 1]; j++) {
				if (AdjListFull_device[j].col == node_2 && AdjListFull_device[j].wt == edge_weight) {
					AdjListFull_device[j].wt = -1;
					//printf("inside del outedge: %d %d %d \n", node_1, node_2, edge_weight);
				}

			}

		}
	}
}

__global__ void insertEdge(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, int inf, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange) {
		//Insertion case
		//for Insertion inst should be 1
		if (allChange_device[index].inst == 1) {
			int node1 = allChange_device[index].node1;
			int node2 = allChange_device[index].node2;
			int edge_weight = allChange_device[index].edge_wt;
			int node_1, node_2;

			//below node_1 node_2 assignment is required for undirected graphs(not required in directed)
			if (SSSP[node2].Dist > SSSP[node1].Dist) {
				node_1 = node1;
				node_2 = node2;
			}
			else {
				node_1 = node2;
				node_2 = node1;
			}
			//printf("inside ins: %d %d %d \n", node_1, node_2, edge_weight);
			//new addition starts(under testing)
			int flag = 1;
			if (SSSP[node_1].Parent == node_2) { flag = 0; } //avoiding 1st type loop creation
			if (SSSP[node_2].Dist == inf && SSSP[node_2].Update == true) { flag = 0; } //avoiding 2nd type loop creation
			//new addition ends(under testing)
			//Check whether node1 is relaxed
			if ((SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight) && flag == 1) {
				//Update Parent and EdgeWt
				SSSP[node_2].Parent = node_1;
				SSSP[node_2].EDGwt = edge_weight;
				SSSP[node_2].Dist = SSSP[node_1].Dist + edge_weight;
				SSSP[node_2].Update = true;
				//Mark Edge to be added
				Edgedone[index] = 1;
				//printf("inside ins if: %d %d \n", node_2, SSSP[node_2].Dist, edge_weight);
			}
		}
	}
}

/*The insertDeleteEdge method might connect wrong edge depending on the sequence when the edge was connected (mainly because of the synchronization related fault)
 We avoid this error by the below method without using locking approach
 This method tries to fit the edges using relax step and if it can fit, a flag is raised.
*/
__global__ void checkInsertedEdges(changeEdge* allChange_device, int totalChange, int* Edgedone, RT_Vertex* SSSP, int* change_d) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange) {
		//Edgedone will be 1 when edge is marked to be inserted
		if (Edgedone[index] == 1) {

			//get the edge
			int node1 = allChange_device[index].node1;
			int node2 = allChange_device[index].node2;
			double edgeWeight = allChange_device[index].edge_wt;
			int node_1, node_2;
			//reset it to 0
			Edgedone[index] = 0;

			//below node_1 node_2 assignment is required for undirected graphs(not required in directed)
			if (SSSP[node2].Dist > SSSP[node1].Dist) {
				node_1 = node1;
				node_2 = node2;
			}
			else {
				node_1 = node2;
				node_2 = node1;
			}

			//***Below two if logic will connect the correct edges.***
			//Check if some other edge was added--mark edge to be added //check x
			if (SSSP[node_2].Dist > SSSP[node_1].Dist + edgeWeight) {
				Edgedone[index] = 1;
			}

			//Check if correct edge wt was written--mark edge to be added //check x
			if ((SSSP[node_2].Parent == node_1) && (SSSP[node_2].EDGwt > edgeWeight)) {
				Edgedone[index] = 1;
			}

			//new addition starts(under testing)
			if (SSSP[node_1].Parent == node_2) { Edgedone[index] = 0; } //avoiding loop creation
			//new addition ends(under testing)

			if (Edgedone[index] == 1) {
				//Update Parent and EdgeWt
				SSSP[node_2].Parent = node_1;
				SSSP[node_2].EDGwt = edgeWeight;
				SSSP[node_2].Dist = SSSP[node_1].Dist + edgeWeight;
				SSSP[node_2].Update = true;
				change_d[0] = 1; //every time node dist is updated, the flag becomes 1
			}
		}
	}
}


/*
updateNeighbors_del function makes dist value of child nodes of a disconnected node to inf
It marks the child nodes also as disconnected nodes
*/
__global__ void updateNeighbors_del(RT_Vertex* SSSP, int nodes, int inf, ColWt* AdjListFull_device, int* AdjListTracker_device, int* change_d) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < nodes && SSSP[index].Dist == inf) {
		for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
			int myn = AdjListFull_device[j].col;
			int mywt = AdjListFull_device[j].wt;
			if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

			if (SSSP[myn].Parent == index && SSSP[myn].Dist != inf) {
				SSSP[myn].Dist = inf;
				SSSP[myn].Update = true;
				//SSSP[myn].EDGwt = inf; //helps to avoid sync error. might be removed
				//new addition
				for (int k = AdjListTracker_device[myn]; k < AdjListTracker_device[myn + 1]; k++) {
					int myn2 = AdjListFull_device[k].col;
					int mywt2 = AdjListFull_device[k].wt;

					if (mywt2 < 0) { continue; } //if mywt = -1, that means edge was deleted
					if (SSSP[myn2].Parent == myn && SSSP[myn2].Dist != inf) {
						SSSP[myn2].Dist = inf;
						SSSP[myn2].Update = true;
						for (int i = AdjListTracker_device[myn2]; i < AdjListTracker_device[myn2 + 1]; i++) {
							int myn3 = AdjListFull_device[i].col;
							int mywt3 = AdjListFull_device[i].wt;

							if (mywt3 < 0) { continue; } //if mywt = -1, that means edge was deleted
							if (SSSP[myn3].Parent == myn2 && SSSP[myn3].Dist != inf) {
								SSSP[myn3].Dist = inf;
								SSSP[myn3].Update = true;
							}
						}
					}
				}
				//new addition ends
				change_d[0] = 1;
			}

		}
	}
}



//1. This method tries to connect the disconnected nodes(disconnected by deletion) with other nodes using the original graph
//2. This method propagates the dist update till the leaf nodes
__global__ void updateNeighbors(RT_Vertex* SSSP, int nodes, int inf, ColWt* AdjListFull_device, int* AdjListTracker_device, int* change_d) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < nodes) {

		//If i is updated--update its neighbors as required
		if (SSSP[index].Update) {
			SSSP[index].Update = false;

			//For neighbor vertices of the affected nodes
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				int myn = AdjListFull_device[j].col;
				int mywt = AdjListFull_device[j].wt;

				if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

				/*if (SSSP[index].Parent == myn && SSSP[myn].Parent == index)
				{
					printf("!!!!loop: %d-%d", index, myn);
				}*/


				//update where parent of myn != index
				if (SSSP[index].Dist > SSSP[myn].Dist + mywt) {
					if (SSSP[myn].Parent != index) {  //avoiding type 1 loop formation
						SSSP[index].Dist = SSSP[myn].Dist + mywt;
						SSSP[index].Update = true;
						SSSP[index].Parent = myn;
						SSSP[index].EDGwt = mywt;
						for (int i = AdjListTracker_device[index]; i < AdjListTracker_device[index + 1]; i++) {
							int myn2 = AdjListFull_device[i].col;
							int mywt2 = AdjListFull_device[i].wt;

							if (mywt2 < 0) { continue; } //if mywt = -1, that means edge was deleted
							if (SSSP[myn2].Parent == index) {
								SSSP[myn2].Dist = SSSP[index].Dist + mywt2;
								SSSP[myn2].Update = true;
								//for (int k = AdjListTracker_device[myn2]; k < AdjListTracker_device[myn2 + 1]; k++) {
								//	int myn3 = AdjListFull_device[k].col;
								//	int mywt3 = AdjListFull_device[k].wt;

								//	if (mywt3 < 0) { continue; } //if mywt = -1, that means edge was deleted
								//	if (SSSP[myn3].Parent == myn2) {
								//		SSSP[myn3].Dist = SSSP[myn2].Dist + mywt3;
								//		SSSP[myn3].Update = true;
								//	}
								//}
							}
						}
						change_d[0] = 1;
						continue;
					}
				}

				////if index node is the parent node of myn, dist of myn is updated even if it increases the dist of myn
				//if (SSSP[myn].Parent == index) { //1st hop search
				//	
				//	SSSP[myn].Dist = SSSP[index].Dist + mywt;
				//	SSSP[myn].Update = true;
				//	//going to 2nd hop
				//	for (int i = AdjListTracker_device[myn]; i < AdjListTracker_device[myn + 1]; i++) {
				//		int myn2 = AdjListFull_device[i].col;
				//		int mywt2 = AdjListFull_device[i].wt;

				//		if (mywt2 < 0) { continue; } //if mywt = -1, that means edge was deleted
				//		if (SSSP[myn2].Parent == myn) {
				//			SSSP[myn2].Dist = SSSP[myn].Dist + mywt2;
				//			SSSP[myn2].Update = true;
				//		}
				//	}
				//	//SSSP[myn].EDGwt = mywt; //helps to avoid sync error. might be removed
				//	change_d[0] = 1;
				//	continue;
				//}


				if (SSSP[myn].Dist > SSSP[index].Dist + mywt) {
					if (SSSP[index].Parent != myn) {
						SSSP[myn].Dist = SSSP[index].Dist + mywt;
						SSSP[myn].Update = true;
						SSSP[myn].Parent = index;
						SSSP[myn].EDGwt = mywt;
						for (int i = AdjListTracker_device[myn]; i < AdjListTracker_device[myn + 1]; i++) {
							int myn2 = AdjListFull_device[i].col;
							int mywt2 = AdjListFull_device[i].wt;

							if (mywt2 < 0) { continue; } //if mywt = -1, that means edge was deleted
							if (SSSP[myn2].Parent == myn) {
								SSSP[myn2].Dist = SSSP[myn].Dist + mywt2;
								SSSP[myn2].Update = true;
							}
						}
						change_d[0] = 1;
					}

				}
			}
		}
	}
}


__global__ void printSSSP(RT_Vertex* SSSP, int nodes) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < 2) {
		int x;
		if (nodes < 40) {
			x = nodes;
		}
		else {
			x = 40;
		}
		printf("from GPU:\n[");
		for (int i = 0; i < x; i++) {
			//cout << "row: " << i << " dist: " << SSSP[i].Dist <<" parent: " << SSSP[i].Parent << endl;
			printf("%d:%d:%d ", i, SSSP[i].Dist, SSSP[i].Parent);
		}
		printf("]\n");
	}
}

#endif