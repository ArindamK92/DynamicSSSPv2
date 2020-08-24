#include <stdio.h>
#include "all_structure_undir.cuh"
#include "gpuFunctions_undir.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<vector>
#include <chrono>
#include <algorithm>

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
nvcc -o op_main CudaSSSPmain.cu
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name
*/
int main(int argc, char* argv[]) {

	int nodes, edges;
	cudaError_t cudaStatus;
	nodes = atoi(argv[2]);
	edges = atoi(argv[3]);
	int deviceId;
	int numberOfSMs;
	int totalInsertion = 0;
	bool zeroDelFlag = false, zeroInsFlag = false;

	//Get gpu device id and number of SMs
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	//Reading Original input graph
	vector<ColWtList> AdjList;
	AdjList.resize(nodes);
	int* AdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
	vector<ColWt> AdjListFull;
	cout << "Reading input graph..." << endl;
	auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	read_graphEdges(AdjList, argv[1], &nodes);
	auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	cout << "Reading input graph completed" << endl;
	cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;

	//Reading change edges input
	//vector<changeEdge> allChange;
	vector<changeEdge> allChange_Ins, allChange_Del;
	cout << "Reading input changed edges data..." << endl;
	auto readCEstartTime = high_resolution_clock::now();//Time calculation starts
	readin_changes(argv[5], allChange_Ins, allChange_Del, AdjList, totalInsertion);
	auto readCEstopTime = high_resolution_clock::now();//Time calculation ends
	auto readCEduration = duration_cast<microseconds>(readCEstopTime - readCEstartTime);// duration calculation
	cout << "Reading input changed edges data completed. totalInsertion:" << totalInsertion << endl;
	cout << "Time taken to read input changed edges: " << readCEduration.count() << " microseconds" << endl;

	//create 1D array from 2D to fit it in GPU
	cout << "creating 1D array from 2D to fit it in GPU" << endl;
	AdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < nodes; i++) {
		AdjListTracker[i + 1] = AdjListTracker[i] + AdjList.at(i).size();
		AdjListFull.insert(std::end(AdjListFull), std::begin(AdjList.at(i)), std::end(AdjList.at(i)));
	}
	cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();
	ColWt* AdjListFull_device;
	cudaStatus = cudaMallocManaged(&AdjListFull_device, (2 * (edges + totalInsertion)) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	std::copy(AdjListFull.begin(), AdjListFull.end(), AdjListFull_device);

	int* AdjListTracker_device;
	cudaStatus = cudaMalloc((void**)&AdjListTracker_device, (nodes + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(AdjListTracker_device, AdjListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

	int totalChangeEdges_Ins = allChange_Ins.size();
	if (totalChangeEdges_Ins == 0) {
		zeroInsFlag = true;
	}
	changeEdge* allChange_Ins_device;
	if(zeroInsFlag != true){
		cudaStatus = cudaMallocManaged(&allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed at allChange_Ins structure");
		}
		std::copy(allChange_Ins.begin(), allChange_Ins.end(), allChange_Ins_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		cudaMemAdvise(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
		//Asynchronous prefetching of data
		cudaMemPrefetchAsync(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), deviceId);
	}



	int totalChangeEdges_Del = allChange_Del.size();
	if (totalChangeEdges_Del == 0) {
		zeroDelFlag = true;
	}
	changeEdge* allChange_Del_device;
	int* counter_del;
	int* affectedNodeList_del;
	int* updatedAffectedNodeList_del;
	int* updated_counter_del;

	if (zeroDelFlag != true) {
		cudaStatus = cudaMallocManaged(&allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed at allChange_Del structure");
		}
		std::copy(allChange_Del.begin(), allChange_Del.end(), allChange_Del_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		cudaMemAdvise(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
		//Asynchronous prefetching of data
		cudaMemPrefetchAsync(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), deviceId);

		counter_del = 0;
		cudaMallocManaged(&counter_del, sizeof(int));
		cudaMallocManaged(&affectedNodeList_del, nodes * sizeof(int));
		cudaMallocManaged(&updatedAffectedNodeList_del, nodes * sizeof(int));
		updated_counter_del = 0;
		cudaMallocManaged(&updated_counter_del, sizeof(int));
	}
	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;


	
	//Reading SSSP Tree input and storing directly on unified memory
	RT_Vertex* SSSP;
	cudaStatus = cudaMallocManaged(&SSSP, nodes * sizeof(RT_Vertex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSP structure");
	}
	cout << "Reading input SSSP tree data..." << endl;
	auto readSSSPstartTime = high_resolution_clock::now();//Time calculation starts
	read_SSSP(SSSP, argv[4], &nodes);
	auto readSSSPstopTime = high_resolution_clock::now();//Time calculation ends
	auto readSSSPduration = duration_cast<microseconds>(readSSSPstopTime - readSSSPstartTime);// duration calculation
	cout << "Reading input SSSP tree data completed" << endl;
	cout << "Time taken to read input input SSSP tree: " << readSSSPduration.count() << " microseconds" << endl;
	//set cudaMemAdviseSetPreferredLocation at GPU for SSSP data
	cudaMemAdvise(SSSP, nodes * sizeof(RT_Vertex), cudaMemAdviseSetPreferredLocation, deviceId);

	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(SSSP, nodes * sizeof(RT_Vertex), deviceId);
	cudaMemPrefetchAsync(AdjListFull_device, edges * sizeof(ColWt), deviceId);

	int* change = 0;
	cudaStatus = cudaMallocManaged(&change, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at change structure");
	}
	cout << "reading input data completed" << endl;



	int* affectedNodeList;
	cudaStatus = cudaMallocManaged(&affectedNodeList, nodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at affectedNodeList structure");
	}
	int* counter = 0;
	cudaStatus = cudaMallocManaged(&counter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at counter structure");
	}
	int* updatedAffectedNodeList_all;
	cudaMallocManaged(&updatedAffectedNodeList_all, nodes * sizeof(int));
	int* updated_counter_all = 0;
	cudaMallocManaged(&updated_counter_all, sizeof(int));



	//processChangedEdges << <(totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, Edgedone, SSSP, totalChangeEdges, inf, AdjListFull_device, AdjListTracker_device, affectedNodeList, counter);
	//    cudaDeviceSynchronize();

	//STEP1
	auto startTime1 = high_resolution_clock::now(); //Time calculation start
	//process change edges
	if (zeroDelFlag != true) {
		deleteEdge << < (totalChangeEdges_Del / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_Del_device, SSSP, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device);
	}
	if (zeroInsFlag != true) {
		insertEdge << < (totalChangeEdges_Ins / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_Ins_device, SSSP, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device);
	}
	//cout << "check b";

	auto stopTime1 = high_resolution_clock::now();//Time calculation ends
	auto duration1 = duration_cast<microseconds>(stopTime1 - startTime1);// duration calculation
	cout << "**Time taken for SSSP update STEP1: "
		<< float(duration1.count()) / 1000 << " milliseconds**" << endl;


	
	
	//STEP2
	auto startTime2 = high_resolution_clock::now();

	if (zeroDelFlag != true) {
		//filter out the nodes affected by edge deletion
		filterAffectedNodes << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, affectedNodeList, counter, nodes, affectedNodeList_del, counter_del);
		cudaDeviceSynchronize();

		*change = 1;
		while (*change == 1) {
			*change = 0;
			//cout << "check d";
			updateNeighbors_del << <(*counter_del / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> >
				(SSSP, updated_counter_del, updatedAffectedNodeList_del, affectedNodeList_del, counter_del, AdjListFull_device, AdjListTracker_device, change);
			cudaDeviceSynchronize();
			*counter_del = *updated_counter_del;
			//cout << "check e";
			copyArray << <(*counter_del / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (updatedAffectedNodeList_del, counter_del, affectedNodeList_del);
			cudaDeviceSynchronize();
			*updated_counter_del = 0;
		}
	}

	auto stopTime2A = high_resolution_clock::now();//Time calculation ends
	auto duration2A = duration_cast<microseconds>(stopTime2A - startTime2);// duration calculation
	cout << "**Time taken for updateNeighbors_del: "
		<< float(duration2A.count()) / 1000 << " milliseconds**" << endl;
	auto startTime2B = high_resolution_clock::now();//Time calculation ends


	filterAllAffected << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, affectedNodeList, counter, nodes);
	cudaDeviceSynchronize();
	int totalAffectedNodes = 0;


	*change = 1;
	while (*change == 1) {
		*change = 0;
		//cout << "check2";
		updateNeighbors << <(*counter / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, counter, affectedNodeList, AdjListFull_device, AdjListTracker_device, change);
		//copyArray << <(*counter / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (updatedAffectedNodeList_all, counter, affectedNodeList);
		//cout << "check3";
		filterAllAffected << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, affectedNodeList, counter, nodes);
		cudaDeviceSynchronize();
		totalAffectedNodes += *counter;
	}
	
	auto stopTime2 = high_resolution_clock::now();//Time calculation ends
	auto duration2B = duration_cast<microseconds>(stopTime2 - startTime2B);// duration calculation
	cout << "**Time taken for updateNeighbors: "
		<< float(duration2B.count()) / 1000 << " milliseconds**" << endl;

	auto duration2 = duration_cast<microseconds>(stopTime2 - startTime2);// duration calculation
	cout << "**Time taken for SSSP update STEP2: "
		<< float(duration2.count()) / 1000 << " milliseconds**" << endl;
	cout << "****Total Time taken for SSSP update: "
		<< (float(duration2.count()) + float(duration1.count())) / 1000 << " milliseconds****" << endl;



	cout << "Total affected nodes: " << totalAffectedNodes;

	cout << "from GPU: \n[";
	printSSSP << <1, 1 >> > (SSSP, nodes);
	cudaDeviceSynchronize();
	int x;
	if (nodes < 40) {
		x = nodes;
	}
	else {
		x = 40;
	}
	cout << "from CPU: \n[";
	for (int i = 0; i < x; i++) {
		cout << i << ":" << SSSP[i].Dist << " ";
	}
	cout << "]\n";
		//print output ends



	if (zeroDelFlag != true) {
		cudaFree(affectedNodeList_del);
		cudaFree(updatedAffectedNodeList_del);
		cudaFree(counter_del);
		cudaFree(updated_counter_del);
		cudaFree(allChange_Del_device);
		
	}
	if (zeroInsFlag != true) {
		cudaFree(allChange_Ins_device);
	}
	cudaFree(change);
	cudaFree(affectedNodeList);
	cudaFree(counter);
	cudaFree(AdjListTracker_device);
	cudaFree(AdjListTracker_device);

	cudaFree(SSSP);
	return 0;
}