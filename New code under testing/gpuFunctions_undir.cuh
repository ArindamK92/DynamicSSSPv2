#ifndef GPUFUNCTIONS_UNDIR_CUH
#define GPUFUNCTIONS_UNDIR_CUH
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



__global__ void processChangedEdges(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, int inf, ColWt* AdjListFull_device, int* AdjListTracker_device, int* changedEdgeList, int* counter)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < totalChange) {
		////Deletion case
		//for Deletion inst should be 0
		if (allChange_device[index].inst == 0) {
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			int edge_weight = allChange_device[index].edge_wt;
			Edgedone[index] = 3;
			//this will check if node1 is parent of node2
			//Mark edge as deleted by making edgewt = inf
			if (SSSP[node_2].Parent == node_1) {
				SSSP[node_2].Dist = inf;
				SSSP[node_2].EDGwt = inf;
				SSSP[node_2].Update = 1;

				//                changedEdgeList[atomicAdd(counter, 1)]=node_2;


				//                *counter=*counter+1;
			}
			else if (SSSP[node_1].Parent == node_2) {
				SSSP[node_1].Dist = inf;
				SSSP[node_1].EDGwt = inf;
				SSSP[node_1].Update = 1;
				//                changedEdgeList[atomicAdd(counter, 1)]=node_1;


				//                *counter=*counter+1;
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

			int flag = 1;
			if (SSSP[node_1].Parent == node_2) { flag = 0; } //avoiding 1st type loop creation
			if (SSSP[node_2].Dist == inf && SSSP[node_2].Update == true) { flag = 0; } //avoiding 2nd type loop creation

			//Check whether node1 is relaxed
			if ((SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight) && flag == 1) {
				//Update Parent and EdgeWt
				SSSP[node_2].Parent = node_1;
				SSSP[node_2].EDGwt = edge_weight;
				SSSP[node_2].Dist = SSSP[node_1].Dist + edge_weight;
				SSSP[node_2].Update = 2;
				//Mark Edge to be added
				Edgedone[index] = 1;
				//                printf("%d\n",index);
				//                changedEdgeList[node_2]=node_2;

				//
				//                *counter=*counter+1;

			}
		}

	}

}


__global__ void filterAffectedNodes(RT_Vertex* SSSP, int* changedEdgesList, int* counter, int nodes, int* changedEdgesList_del, int* counter_del)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nodes) {
		//		if (SSSP[i].Update == 2) {
		////			printf("counter %d=\n", *counter);
		//			changedEdgesList[atomicAdd(counter, 1)] = i;
		//		}
		if (SSSP[i].Update == 1) {
			//			printf("counter %d=\n", *counter);
			changedEdgesList_del[atomicAdd(counter_del, 1)] = i;
			//			changedEdgesList[atomicAdd(counter, 1)] = i;
		}

	}

}



__global__ void filterAllAffected(RT_Vertex* SSSP, int* changedEdgesList, int* counter, int nodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nodes) {
		if (SSSP[i].Update == 2 || SSSP[i].Update == 1) {
			//			printf("counter %d=\n", *counter);
			changedEdgesList[atomicAdd(counter, 1)] = i;
		}


	}

}




__global__ void deleteEdge(changeEdge* allChange_Del_device, RT_Vertex* SSSP, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChangeEdges_Del) {
		////Deletion case
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
		int edge_weight = allChange_Del_device[index].edge_wt;

		//this will check if node1 is parent of node2
		//Mark edge as deleted by making edgewt = inf
		if (SSSP[node_2].Parent == node_1) {
			SSSP[node_2].Dist = 999999;
			//SSSP[node_2].EDGwt = inf;
			SSSP[node_2].Update = 1;
		}
		else if (SSSP[node_1].Parent == node_2) {
			SSSP[node_1].Dist = 999999;
			//SSSP[node_1].EDGwt = inf;
			SSSP[node_1].Update = 1;
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

__global__ void insertEdge(changeEdge* allChange_Ins_device, RT_Vertex* SSSP, int totalChangeEdges_Ins, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChangeEdges_Ins) {
		//Insertion cas
		int node1 = allChange_Ins_device[index].node1;
		int node2 = allChange_Ins_device[index].node2;
		int edge_weight = allChange_Ins_device[index].edge_wt;
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

		int flag = 1;
		if (SSSP[node_1].Parent == node_2) { flag = 0; } //avoiding 1st type loop creation
		if (SSSP[node_2].Dist == 999999 && SSSP[node_2].Update == 1) { flag = 0; } //avoiding 2nd type loop creation

		//Check whether node1 is relaxed
		if ((SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight) && flag == 1) {
			//Update Parent and EdgeWt
			SSSP[node_2].Parent = node_1;
			SSSP[node_2].EDGwt = edge_weight;
			SSSP[node_2].Dist = SSSP[node_1].Dist + edge_weight;
			SSSP[node_2].Update = 2;
			//Mark Edge to be added
			//Edgedone[index] = 1;
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
			//Check if some other edge was added--mark edge to be added
			if (SSSP[node_2].Dist > SSSP[node_1].Dist + edgeWeight) {
				Edgedone[index] = 1;
			}

			//Check if correct edge wt was written--mark edge to be added
			if ((SSSP[node_2].Parent == node_1) && (SSSP[node_2].EDGwt > edgeWeight)) {
				Edgedone[index] = 1;
			}

			if (SSSP[node_1].Parent == node_2) { Edgedone[index] = 0; } //avoiding loop creation

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

__global__ void updateNeighbors_del(RT_Vertex* SSSP, int* updated_counter_del, int* updatedEdgesList_del, int* changedEdgesList_del, int* counter_del, ColWt* AdjListFull_device, int* AdjListTracker_device, int* change) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;


	if (index < *counter_del && SSSP[changedEdgesList_del[index]].Dist == 999999) {
		for (int j = AdjListTracker_device[changedEdgesList_del[index]]; j < AdjListTracker_device[changedEdgesList_del[index] + 1]; j++) {
			int myn = AdjListFull_device[j].col;
			int mywt = AdjListFull_device[j].wt;
			if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

			if (SSSP[myn].Parent == changedEdgesList_del[index] && SSSP[myn].Dist != 999999) {
				SSSP[myn].Dist = 999999;
				SSSP[myn].Update = 1;

				updatedEdgesList_del[atomicAdd(updated_counter_del, 1)] = myn;
				//				changedEdgesList[atomicAdd(counter, 1)] = myn;
				*change = 1;
			}

		}
	}
}
__global__ void copyArray(int* updatedEdgesList_del, int* counter_del, int* changedEdgesList_del) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < *counter_del) {
		changedEdgesList_del[index] = updatedEdgesList_del[index];

	}
}


//1. This method tries to connect the disconnected nodes(disconnected by deletion) with other nodes using the original graph
//2. This method propagates the dist update till the leaf nodes


__global__ void updateNeighbors(RT_Vertex* SSSP, int* counter, int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* change) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < *counter) {

		//If i is updated--update its neighbors as required

		int tmpValueforIndex = affectedNodeList[index];
		SSSP[tmpValueforIndex].Update = 0;

		//For neighbor vertices of the affected nodes
		for (int j = AdjListTracker_device[tmpValueforIndex]; j < AdjListTracker_device[tmpValueforIndex + 1]; j++) {
			int myn = AdjListFull_device[j].col;
			int mywt = AdjListFull_device[j].wt;

			if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

			//update where parent of myn != index
			if (SSSP[tmpValueforIndex].Dist > SSSP[myn].Dist + mywt) {
				if (SSSP[myn].Parent != tmpValueforIndex) {  //avoiding type 1 loop formation
					SSSP[tmpValueforIndex].Dist = SSSP[myn].Dist + mywt;
					SSSP[tmpValueforIndex].Update = 2;
					SSSP[tmpValueforIndex].Parent = myn;
					*change = 1;
					continue;
				}
			}

			if (SSSP[myn].Dist > SSSP[tmpValueforIndex].Dist + mywt) {
				if (SSSP[tmpValueforIndex].Parent != myn) {//avoiding type 1 loop formation
					SSSP[myn].Dist = SSSP[tmpValueforIndex].Dist + mywt;
					SSSP[myn].Update = 2;
					SSSP[myn].Parent = tmpValueforIndex;
					*change = 1;
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
			printf("%d:%d:%d ", i, SSSP[i].Dist, SSSP[i].Parent);
		}
		printf("]\n");
	}
}

#endif