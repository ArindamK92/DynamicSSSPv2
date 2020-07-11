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

__global__ void initializeEdgedone(int* Edgedone, int totalChange)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange)
	{
		Edgedone[index] = -1;
	}
}

__global__ void insertDeleteEdge(changeEdge* allChange_device, int* Edgedone, RT_Vertex* SSSP, int totalChange, double inf, ColWt* InEdgesListFull_device, ColWt* OutEdgesListFull_device, int* InEdgesListTracker_device, int* OutEdgesListTracker_device)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange)
	{
		int node_1 = allChange_device[index].node1;
		int node_2 = allChange_device[index].node2;
		int edge_weight = allChange_device[index].edge_wt;

		////Deletion case
		if (allChange_device[index].inst == 0 /*&& Edgedone[index] != 0*/)  //for Deletion inst should be 0
		{
			Edgedone[index] = 3;
			bool iskeyedge = false;
			//this will check if node1 is parent of node2
			//Mark edge as deleted by making edgewt = inf
			if (SSSP[node_2].Parent == node_1)
			{
				SSSP[node_2].EDGwt = inf;
				SSSP[node_2].Update = true;
				iskeyedge = true;
			}
			//If  Key Edge is Deleted Set weights to -1 in input graph 
			if (iskeyedge)
			{
				//mark the edge as deleted in inEdges list
				for (int j = InEdgesListTracker_device[node_2]; j < InEdgesListTracker_device[node_2 + 1]; j++)
				{
					if (InEdgesListFull_device[j].col == node_1 && InEdgesListFull_device[j].wt == edge_weight)
					{
						InEdgesListFull_device[j].wt = -1;
					}

				}
				//mark the edge as deleted in outEdges list
				for (int j = OutEdgesListTracker_device[node_1]; j < OutEdgesListTracker_device[node_1 + 1]; j++)
				{
					if (OutEdgesListFull_device[j].col == node_2 && OutEdgesListFull_device[j].wt == edge_weight)
					{
						OutEdgesListFull_device[j].wt = -1;
					}

				}
			}

		}
		//Insertion case
		if (allChange_device[index].inst == 1)  //for Insertion inst should be 1
		{
				//Check whether node1 is relaxed
				if (SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight)
				{
					//Update Parent and EdgeWt
					SSSP[node_2].Parent = node_1;
					SSSP[node_2].EDGwt = edge_weight;
					SSSP[node_2].Dist = SSSP[node_1].Dist + edge_weight;
					SSSP[node_2].Update = true;
					//Mark Edge to be added
					Edgedone[index] = 1;
				}
		}
	}
}


/*The insertDeleteEdge method might connect wrong edge depending on the sequence when the edge was connected (mainly because of the synchronization related fault)
 We avoid this error by the below method without using locking approach
 This method tries to fit the edges using relax step and if it can fit, a flag is raised. 
*/
__global__ void checkInsertedEdges(changeEdge* allChange_device, int totalChange, int* Edgedone, RT_Vertex* SSSP, int* change_d)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalChange)
	{

		if (Edgedone[index] == 1) //Edgedone will be 1 when edge is marked to be inserted
		{

			//get the edge
			int node_1 = allChange_device[index].node1;
			int node_2 = allChange_device[index].node2;
			double edgeWeight = allChange_device[index].edge_wt;
			//reset it to 0
			Edgedone[index] = 0;

			//***Below two if logic will connect the correct edges.***
			//Check if some other edge was added--mark edge to be added //check x
			if (SSSP[node_2].Dist > SSSP[node_1].Dist + edgeWeight)
			{
				Edgedone[index] = 1;
			}

			//Check if correct edge wt was written--mark edge to be added //check x
			if ((SSSP[node_2].Parent == node_1) && (SSSP[node_2].EDGwt > edgeWeight))
			{
				Edgedone[index] = 1;
			}


			if (Edgedone[index] == 1)
			{
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


//void processCE(int deviceId, int totalChange, changeEdge* allChange_device, RT_Vertex* SSSP, ColWt* InEdgesListFull_device, ColWt* OutEdgesListFull_device, int* InEdgesListTracker_device, int* OutEdgesListTracker_device)
//{
//	cudaError_t cudaStatus;
//	double inf = std::numeric_limits<double>::infinity();
//	int* Edgedone;
//	cudaMallocManaged(&Edgedone, (totalChange) * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed at SSSP structure");
//	}
//	cudaMemPrefetchAsync(Edgedone, (totalChange) * sizeof(int), deviceId);
//	//initialize Edgedone array with -1
//	initializeEdgedone << <(totalChange / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (Edgedone, totalChange);
//	insertDeleteEdge << < (totalChange / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, Edgedone, SSSP, totalChange, inf, InEdgesListFull_device, OutEdgesListFull_device, InEdgesListTracker_device, OutEdgesListTracker_device);
//	cudaDeviceSynchronize();
//	
//	cudaFree(Edgedone);
//}


#endif