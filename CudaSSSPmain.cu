#include <stdio.h>
#include "all_structure.cuh"
#include "gpuFunctions.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<vector>
#include <chrono> 


#define THREADS_PER_BLOCK 1024 //we can change it

using namespace std;
using namespace std::chrono;


/*
1st arg: original graph file name
2nd arg: no. of nodes     
3rd arg: no. of edges    
4th arg: input SSSP file name
5th arg: change edges file name
****main commands to run****
nvcc -o op main2.cu
./op <fullgraph file name> <SSSP file name> <changeEdges file name> <no. of nodes> <no. of edges * 2 (or total number of lines in fullgraph file)>
*/
int main(int argc, char* argv[]) {

	int nodes, edges;
	cudaError_t cudaStatus;
	nodes = atoi(argv[2]);
	edges = atoi(argv[3]);
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	
	//Reading Original input graph
	vector<ColWtList> InEdgesList;
	InEdgesList.resize(nodes);
	int* InEdgesListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row 
	vector<ColWt> InEdgesListFull;
	vector<ColWtList> OutEdgesList;
	OutEdgesList.resize(nodes);
	int* OutEdgesListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row 
	vector<ColWt> OutEdgesListFull;
	read_graphEdges(InEdgesList, argv[1], &nodes,  OutEdgesList);
	//Reading change edges input
	vector<changeEdge> allChange;
	readin_changes(argv[5], allChange, InEdgesList, OutEdgesList);

	//create 1D array from 2D to fit it in GPU
	InEdgesListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	OutEdgesListTracker[0] = 0; //start pointer points to the first index of OutEdgesList
	for (int i = 0; i < nodes; i++)
	{
		InEdgesListTracker[i + 1] = InEdgesListTracker[i] + InEdgesList.at(i).size();
		InEdgesListFull.insert(std::end(InEdgesListFull), std::begin(InEdgesList.at(i)), std::end(InEdgesList.at(i)));
		OutEdgesListTracker[i + 1] = OutEdgesListTracker[i] + OutEdgesList.at(i).size();
		OutEdgesListFull.insert(std::end(OutEdgesListFull), std::begin(OutEdgesList.at(i)), std::end(OutEdgesList.at(i)));
	}



	//Transferring input graph and change edges data to GPU
	ColWt* InEdgesListFull_device;
	cudaStatus = cudaMallocManaged(&InEdgesListFull_device, edges * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	std::copy(InEdgesListFull.begin(), InEdgesListFull.end(), InEdgesListFull_device);

	ColWt* OutEdgesListFull_device;
	cudaStatus = cudaMallocManaged(&OutEdgesListFull_device, edges * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	std::copy(OutEdgesListFull.begin(), OutEdgesListFull.end(), OutEdgesListFull_device);
	
	int* InEdgesListTracker_device;
	cudaStatus = cudaMalloc((void**)&InEdgesListTracker_device, nodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(InEdgesListTracker_device, InEdgesListTracker, nodes * sizeof(int), cudaMemcpyHostToDevice);
	int* OutEdgesListTracker_device;
	cudaStatus = cudaMalloc((void**)&OutEdgesListTracker_device, nodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at OutEdgesListTracker_device");
	}
	cudaMemcpy(OutEdgesListTracker_device, OutEdgesListTracker, nodes * sizeof(int), cudaMemcpyHostToDevice);
	
	int totalChangeEdges = allChange.size();
	changeEdge* allChange_device;
	cudaStatus = cudaMallocManaged(&allChange_device, totalChangeEdges * sizeof(changeEdge));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at allChange structure");
	}
	std::copy(allChange.begin(), allChange.end(), allChange_device);
	//set cudaMemAdviseSetReadMostly by the GPU for change edge data
	cudaMemAdvise(allChange_device, totalChangeEdges * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);


	//Test start
	/*cout << "change edges:" << endl;
	for (int i = 0; i < totalChangeEdges; i++)
	{
		cout << allChange_device[i].node1 << " " << allChange_device[i].node2 << " " << allChange_device[i].edge_wt << " " << allChange_device[i].inst << endl;
	}
	cout << "In edges in Unified memory" << endl;
	for (int i = 0; i < nodes; i++)
	{
		cout << "row: " << i << endl;
		for (int j = InEdgesListTracker[i]; j < InEdgesListTracker[i + 1]; j++)
		{
			cout << InEdgesListFull_device[j].col << " : " << InEdgesListFull_device[j].wt << endl;
		}
	}
	cout << "Out edges in Unified memory" << endl;
	for (int i = 0; i < nodes; i++)
	{
		cout << "row: " << i << endl;
		for (int j = OutEdgesListTracker[i]; j < OutEdgesListTracker[i + 1]; j++)
		{
			cout << OutEdgesListFull_device[j].col << " : " << OutEdgesListFull_device[j].wt << endl;
		}
	}*/
	//Test end
	
	//Reading SSSP Tree input and storing directly on unified memory
	RT_Vertex* SSSP;
	cudaStatus = cudaMallocManaged(&SSSP, nodes * sizeof(RT_Vertex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSP structure");
	}
	read_SSSP(SSSP, argv[4], &nodes);
	//set cudaMemAdviseSetPreferredLocation at GPU for SSSP data
	cudaMemAdvise(SSSP, nodes * sizeof(RT_Vertex), cudaMemAdviseSetPreferredLocation, deviceId);



	//test start
	/*cout << "after reading SSSP:" << endl;
	for (int i = 0; i < nodes; i++)
	{
		cout << "row: " << i << " dist: " << SSSP[i].Dist << " parent: " << SSSP[i].Parent << endl;

	}*/
	//test end



	//double inf = std::numeric_limits<double>::infinity();
	int inf = 999999;
	int* Edgedone;
	cudaMallocManaged(&Edgedone, (totalChangeEdges) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSP structure");
	}
	//set cudaMemAdviseSetPreferredLocation at GPU for SSSP data
	cudaMemAdvise(Edgedone, (totalChangeEdges) * sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);
	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(Edgedone, (totalChangeEdges) * sizeof(int), deviceId);
	cudaMemPrefetchAsync(allChange_device, totalChangeEdges * sizeof(changeEdge), deviceId);
	cudaMemPrefetchAsync(SSSP, nodes * sizeof(RT_Vertex), deviceId);
	cudaMemPrefetchAsync(InEdgesListFull_device, edges * sizeof(ColWt), deviceId);
	cudaMemPrefetchAsync(OutEdgesListFull_device, edges * sizeof(ColWt), deviceId);
	//initialize Edgedone array with -1
	initializeEdgedone << <(totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (Edgedone, totalChangeEdges);
	deleteEdge << < (totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, Edgedone, SSSP, totalChangeEdges, inf, InEdgesListFull_device, OutEdgesListFull_device, InEdgesListTracker_device, OutEdgesListTracker_device);
	insertEdge << < (totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, Edgedone, SSSP, totalChangeEdges, inf, InEdgesListFull_device, OutEdgesListFull_device, InEdgesListTracker_device, OutEdgesListTracker_device);
	
	
	
	//test start
	/*cudaDeviceSynchronize();
	cout << "\nafter insertDeleteEdge SSSP:" << endl;
	
	for (int i = 0; i < nodes; i++)
	{
		cout << "row: " << i << " dist: " << SSSP[i].Dist << " parent: " << SSSP[i].Parent << endl;

	}*/
	//test end





	//Go over the inserted edges to see if they need to be changed. Correct edges are connected in this stage
	int* change_d = new int[1];
	int* change = new int[1];
	change[0] = 1;
	cudaMalloc((void**)&change_d, 1 * sizeof(int));
	while (change[0] == 1)
	{
		change[0] = 0;
		cudaMemcpy(change_d, change, 1 * sizeof(int), cudaMemcpyHostToDevice);
		checkInsertedEdges << < (totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, totalChangeEdges, Edgedone, SSSP, change_d);
		cudaMemcpy(change, change_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

	}

	cudaFree(Edgedone); //free memory before neighbor update

	int its = 0;
	change[0] = 1;
	while (change[0] == 1 && its < 70)
	{
		//printf("Iteration:%d \n", its);

		change[0] = 0;
		cudaMemcpy(change_d, change, 1 * sizeof(int), cudaMemcpyHostToDevice);
		updateNeighbors << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, nodes, inf, InEdgesListFull_device, OutEdgesListFull_device, InEdgesListTracker_device, OutEdgesListTracker_device, change_d);
		cudaMemcpy(change, change_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		its++;
	}//end of while
	printf("Total Iterations to Converge %d \n", its);

	for (int i = 0; i < nodes; i++)
	{
		cout << "row: " << i << " dist: " << SSSP[i].Dist <<" parent: " << SSSP[i].Parent << endl;
		
	}



	cudaFree(change_d);
	cudaFree(InEdgesListTracker_device);
	cudaFree(OutEdgesListTracker_device);
	cudaFree(OutEdgesListFull_device);
	cudaFree(InEdgesListFull_device);
	cudaFree(allChange_device);
	cudaFree(SSSP);
	return 0;
}


