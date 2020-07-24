#ifndef GPUFUNCTIONS_CUH
#define GPUFUNCTIONS_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "all_structure.cuh"
using namespace std;

#define THREADS_PER_BLOCK 1024 //we can change it

__global__ void initializeEdgedone(int* Edgedone, int totalChange){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange)
	{
		Edgedone[index] = -1;
	}
}

__global__ void deleteEdge(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, int inf, ColWt* InEdgesListFull_device, ColWt* OutEdgesListFull_device, int* InEdgesListTracker_device, int* OutEdgesListTracker_device){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange){
		////Deletion case
		//for Deletion inst should be 0
		if (allChange_device[index].inst == 0)  {
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			int edge_weight = allChange_device[index].edge_wt;
			Edgedone[index] = 3;
			//bool iskeyedge = false;
			//this will check if node1 is parent of node2
			//Mark edge as deleted by making edgewt = inf
			if (SSSP[node_2].Parent == node_1){
				SSSP[node_2].Dist = inf;
				SSSP[node_2].EDGwt = inf;
				SSSP[node_2].Update = true;
				//new addition starts(under testing)
				//SSSP[node_2].Parent = node_2; //marked disconnected node's parent as the same node
				//new addition ends(under testing)
				//iskeyedge = true;
				//printf("inside del: %d %d \n", node_2, SSSP[node_2].EDGwt, edge_weight);
			}
			//If  Key Edge is Deleted Set weights to -1 in input graph 
			//if (iskeyedge)
			//{
				//mark the edge as deleted in inEdges list
				for (int j = InEdgesListTracker_device[node_2]; j < InEdgesListTracker_device[node_2 + 1]; j++){
					if (InEdgesListFull_device[j].col == node_1 && InEdgesListFull_device[j].wt == edge_weight){
						InEdgesListFull_device[j].wt = -1;
						//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
					}

				}
				//mark the edge as deleted in outEdges list
				for (int j = OutEdgesListTracker_device[node_1]; j < OutEdgesListTracker_device[node_1 + 1]; j++){
					if (OutEdgesListFull_device[j].col == node_2 && OutEdgesListFull_device[j].wt == edge_weight){
						OutEdgesListFull_device[j].wt = -1;
						//printf("inside del outedge: %d %d %d \n", node_1, node_2, edge_weight);
					}

				}
			//}

		}
	}
}

__global__ void insertEdge(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, int inf, ColWt* InEdgesListFull_device, ColWt* OutEdgesListFull_device, int* InEdgesListTracker_device, int* OutEdgesListTracker_device){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange){
		//Insertion case
		//for Insertion inst should be 1
		if (allChange_device[index].inst == 1){
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			int edge_weight = allChange_device[index].edge_wt;
			//printf("inside ins: %d %d %d \n", node_1, node_2, edge_weight);
			//new addition starts(under testing)
			int flag = 1;
			if (SSSP[node_1].Parent == node_2) { flag = 0; } //avoiding 1st type loop creation
			if (SSSP[node_2].Dist == inf && SSSP[node_2].Update == true) { flag = 0; } //avoiding 2nd type loop creation
			//new addition ends(under testing)
			//Check whether node1 is relaxed
			if ((SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight) && flag == 1){
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
__global__ void checkInsertedEdges(changeEdge* allChange_device, int totalChange, int* Edgedone, RT_Vertex* SSSP, int* change_d){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange){
		//Edgedone will be 1 when edge is marked to be inserted
		if (Edgedone[index] == 1){

			//get the edge
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			double edgeWeight = allChange_device[index].edge_wt;
			//reset it to 0
			Edgedone[index] = 0;

			//***Below two if logic will connect the correct edges.***
			//Check if some other edge was added--mark edge to be added //check x
			if (SSSP[node_2].Dist > SSSP[node_1].Dist + edgeWeight){
				Edgedone[index] = 1;
			}

			//Check if correct edge wt was written--mark edge to be added //check x
			if ((SSSP[node_2].Parent == node_1) && (SSSP[node_2].EDGwt > edgeWeight)){
				Edgedone[index] = 1;
			}

			//new addition starts(under testing)
			if (SSSP[node_1].Parent == node_2) { Edgedone[index] = 0; } //avoiding loop creation
			//new addition ends(under testing)

			if (Edgedone[index] == 1){
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
__global__ void updateNeighbors_del(RT_Vertex* SSSP, int nodes, int inf, ColWt* OutEdgesListFull_device, int* OutEdgesListTracker_device, int* change_d){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < nodes && SSSP[index].Dist == inf){
			for (int j = OutEdgesListTracker_device[index]; j < OutEdgesListTracker_device[index + 1]; j++){
				int myn = OutEdgesListFull_device[j].col;
				int mywt = OutEdgesListFull_device[j].wt;
				if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted
				
				if (SSSP[myn].Parent == index && SSSP[myn].Dist != inf){
					SSSP[myn].Dist = inf;
					SSSP[myn].Update = true;
					//SSSP[myn].EDGwt = inf; //helps to avoid sync error. might be removed
					change_d[0] = 1;
				}
				
			}
	}
}



//1. This method tries to connect the disconnected nodes(disconnected by deletion) with other nodes using the original graph
//2. This method propagates the dist update till the leaf nodes
__global__ void updateNeighbors(RT_Vertex* SSSP, int nodes, int inf, ColWt* InEdgesListFull_device, ColWt* OutEdgesListFull_device, int* InEdgesListTracker_device, int* OutEdgesListTracker_device, int* change_d){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < nodes){

		//If i is updated--update its neighbors as required
		if (SSSP[index].Update){
			SSSP[index].Update = false;

			//For incoming edges of the affected nodes
			for (int j = InEdgesListTracker_device[index]; j < InEdgesListTracker_device[index + 1]; j++){
				int myn = InEdgesListFull_device[j].col;
				int mywt = InEdgesListFull_device[j].wt;

				if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted
				//new addition starts
				/*if (SSSP[index].Parent == myn && SSSP[myn].Parent == index)
				{
					printf("!!!!loop: %d-%d", index, myn);
				}*/
				//new addition ends
				if(SSSP[myn].Parent == index) { continue; } //avoiding loop formation

				//update where parent of myn != index
				if (SSSP[index].Dist > SSSP[myn].Dist + mywt) {
					SSSP[index].Dist = SSSP[myn].Dist + mywt;
					SSSP[index].Update = true;
					SSSP[index].Parent = myn;
					SSSP[index].EDGwt = mywt;
					change_d[0] = 1;
					continue;
				}
			}
			for (int j = OutEdgesListTracker_device[index]; j < OutEdgesListTracker_device[index + 1]; j++){
				int myn = OutEdgesListFull_device[j].col;
				int mywt = OutEdgesListFull_device[j].wt;
				if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted
				//new addition starts(under testing)
				/*if (SSSP[index].Parent == myn && SSSP[myn].Parent == index)
				{
					printf("!!!!loop: %d-%d", index, myn);
				}*/
				if (SSSP[index].Parent == myn) { continue; }
				//new addition ends(under testing)
				//if index node is the parent node of myn, dist of myn is updated even if it increases the dist of myn
				if (SSSP[myn].Parent == index){
					//in case of disconnected index node due to deletion
					if (SSSP[index].Dist == inf) {
						SSSP[myn].Dist = inf;
					}
					else { //when the dist of index increases due to reconnecting the disconnected subgraphs
						SSSP[myn].Dist = SSSP[index].Dist + mywt;
					}
					SSSP[myn].Update = true;
					//SSSP[myn].Parent = index; //parent of myn is already the index node
					SSSP[myn].EDGwt = mywt; //helps to avoid sync error. might be removed
					change_d[0] = 1;
					continue;
				}

				//update both cases where parent of myn == index or parent of myn != index
				if (SSSP[myn].Dist > SSSP[index].Dist + mywt) {
					SSSP[myn].Dist = SSSP[index].Dist + mywt;
					SSSP[myn].Update = true;
					SSSP[myn].Parent = index;
					SSSP[myn].EDGwt = mywt;
					change_d[0] = 1;
				}
			}
		}
	}
}


__global__ void printSSSP(RT_Vertex* SSSP, int nodes){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < 2){
		int x;
		if (nodes < 40){
			x = nodes;
		}
		else {
			x = 40;
		}
		printf("from GPU:\n[");
		for (int i = 0; i < x; i++){
			//cout << "row: " << i << " dist: " << SSSP[i].Dist <<" parent: " << SSSP[i].Parent << endl;
			printf("%d:%d:%d ", i, SSSP[i].Dist, SSSP[i].Parent);
		}
		printf("]\n");
	}
}

#endif