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

	//Get gpu device id and number of SMs
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
	cout << "Reading input graph..." << endl;
	auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	read_graphEdges(InEdgesList, argv[1], &nodes,  OutEdgesList);
	auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	cout << "Reading input graph completed" << endl;
	cout << "Time taken to read input graph: "<< readGraphduration.count() << " microseconds" << endl;
	
	//Reading change edges input
	vector<changeEdge> allChange;
	cout << "Reading input changed edges data..." << endl;
	auto readCEstartTime = high_resolution_clock::now();//Time calculation starts
	readin_changes(argv[5], allChange, InEdgesList, OutEdgesList, totalInsertion);
	auto readCEstopTime = high_resolution_clock::now();//Time calculation ends
	auto readCEduration = duration_cast<microseconds>(readCEstopTime - readCEstartTime);// duration calculation
	cout << "Reading input changed edges data completed. totalInsertion:" << totalInsertion << endl;
	cout << "Time taken to read input changed edges: " << readCEduration.count() << " microseconds" << endl;

	//create 1D array from 2D to fit it in GPU
	cout << "creating 1D array from 2D to fit it in GPU" << endl;
	InEdgesListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	OutEdgesListTracker[0] = 0; //start pointer points to the first index of OutEdgesList
	for (int i = 0; i < nodes; i++){
		InEdgesListTracker[i + 1] = InEdgesListTracker[i] + InEdgesList.at(i).size();
		InEdgesListFull.insert(std::end(InEdgesListFull), std::begin(InEdgesList.at(i)), std::end(InEdgesList.at(i)));
		OutEdgesListTracker[i + 1] = OutEdgesListTracker[i] + OutEdgesList.at(i).size();
		OutEdgesListFull.insert(std::end(OutEdgesListFull), std::begin(OutEdgesList.at(i)), std::end(OutEdgesList.at(i)));
	}
	cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();
	//cout << "Transferring incoming edges data to GPU" << endl;
	ColWt* InEdgesListFull_device;
	cudaStatus = cudaMallocManaged(&InEdgesListFull_device, (edges + totalInsertion) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	//cout << "Transferring incoming edges data to GPU before copy. size:" << InEdgesListFull.size() << endl;
	/*for (int i = 0; i < InEdgesListFull.size(); i++)
	{
		cout << InEdgesListFull.at(i).col << ":" << InEdgesListFull.at(i).wt << endl;
	}*/
	std::copy(InEdgesListFull.begin(), InEdgesListFull.end(), InEdgesListFull_device);

	//cout << "Transferring outgoing edges data to GPU" << endl;
	ColWt* OutEdgesListFull_device;
	cudaStatus = cudaMallocManaged(&OutEdgesListFull_device, (edges + totalInsertion) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	std::copy(OutEdgesListFull.begin(), OutEdgesListFull.end(), OutEdgesListFull_device);
	
	//cout << "Transferring incoming edges tracker to GPU" << endl;
	int* InEdgesListTracker_device;
	cudaStatus = cudaMalloc((void**)&InEdgesListTracker_device, (nodes+1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(InEdgesListTracker_device, InEdgesListTracker, (nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
	
	//cout << "Transferring outgoing edges tracker to GPU" << endl;
	int* OutEdgesListTracker_device;
	cudaStatus = cudaMalloc((void**)&OutEdgesListTracker_device, (nodes+1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at OutEdgesListTracker_device");
	}
	cudaMemcpy(OutEdgesListTracker_device, OutEdgesListTracker, (nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
	
	//cout << "Transferring change edges data to GPU" << endl;
	int totalChangeEdges = allChange.size();
	changeEdge* allChange_device;
	cudaStatus = cudaMallocManaged(&allChange_device, totalChangeEdges * sizeof(changeEdge));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at allChange structure");
	}
	std::copy(allChange.begin(), allChange.end(), allChange_device);

	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;


	//set cudaMemAdviseSetReadMostly by the GPU for change edge data
	//cout << "setting GPU advice for change edges" << endl;
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
	cout << "Reading input SSSP tree data..." << endl;
	auto readSSSPstartTime = high_resolution_clock::now();//Time calculation starts
	read_SSSP(SSSP, argv[4], &nodes);
	auto readSSSPstopTime = high_resolution_clock::now();//Time calculation ends
	auto readSSSPduration = duration_cast<microseconds>(readSSSPstopTime - readSSSPstartTime);// duration calculation
	cout << "Reading input SSSP tree data completed" << endl;
	cout << "Time taken to read input input SSSP tree: " << readSSSPduration.count() << " microseconds" << endl;
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
	int* change_d = new int[1];
	int* change = new int[1];
	change[0] = 1;
	cudaMalloc((void**)&change_d, 1 * sizeof(int));
	int its = 0;
	cout << "reading input data completed" << endl;


	
	auto startTime1 = high_resolution_clock::now(); //Time calculation start
	//initialize Edgedone array with -1
	initializeEdgedone << <(totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (Edgedone, totalChangeEdges);
	//process change edges
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
	while (change[0] == 1){
		change[0] = 0;
		cudaMemcpy(change_d, change, 1 * sizeof(int), cudaMemcpyHostToDevice);
		checkInsertedEdges << < (totalChangeEdges / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (allChange_device, totalChangeEdges, Edgedone, SSSP, change_d);
		cudaMemcpy(change, change_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

	}
	cudaFree(Edgedone); //free memory before neighbor update
	auto stopTime1 = high_resolution_clock::now();//Time calculation ends
	auto duration1 = duration_cast<microseconds>(stopTime1 - startTime1);// duration calculation
	cout << "**Time taken for STEP 1: "
		<< float(duration1.count())/1000 << " milliseconds**" << endl;


	//Step 2 starts
	auto startTime2 = high_resolution_clock::now(); //Time calculation start
	change[0] = 1;
	while (change[0] == 1 /*&& its < 202*/){
		//printf("Iteration:%d \n", its);
		change[0] = 0;
		cudaMemcpy(change_d, change, 1 * sizeof(int), cudaMemcpyHostToDevice);
		updateNeighbors_del << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, nodes, inf, OutEdgesListFull_device, OutEdgesListTracker_device, change_d);
		cudaMemcpy(change, change_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//its++;
		//cout << "itr:" << its << " " << endl;
	}
	/*its = 0;*/
	//new addition ends




	//update the distance of neighbors and connect the disconnected subgraphs
	change[0] = 1;
	while (change[0] == 1 /*&& its < 200*/){
		change[0] = 0;
		cudaMemcpy(change_d, change, 1 * sizeof(int), cudaMemcpyHostToDevice);
		updateNeighbors << <(nodes / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, nodes, inf, InEdgesListFull_device, OutEdgesListFull_device, InEdgesListTracker_device, OutEdgesListTracker_device, change_d);
		cudaMemcpy(change, change_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		its++;
	}
	auto stopTime2 = high_resolution_clock::now();//Time calculation ends
	auto duration2 = duration_cast<microseconds>(stopTime2 - startTime2);// duration calculation
	cout << "**Time taken for STEP 2: "
		<< float(duration2.count())/1000 << " milliseconds**" << endl;
	printf("Total Iterations to Converge %d \n", its);
	cout << "****Total Time taken for SSSP update: "
		<< float(duration1.count() + duration2.count()) / 1000 << " milliseconds****" << endl;


	//print output:
	printSSSP << <1,1>> > (SSSP, nodes);
	cudaDeviceSynchronize();
	int x;
	if (nodes < 40){
		x = nodes;
	}
	else {
		x = 40;
	}
	cout << "from CPU: \n[";
	for (int i = 0; i < x; i++){
		//cout << "row: " << i << " dist: " << SSSP[i].Dist <<" parent: " << SSSP[i].Parent << endl;
		cout << i << ":" << SSSP[i].Dist << " ";
	}
	cout << "]\n";
	//print output ends



	cudaFree(change_d);
	cudaFree(InEdgesListTracker_device);
	cudaFree(OutEdgesListTracker_device);
	cudaFree(OutEdgesListFull_device);
	cudaFree(InEdgesListFull_device);
	cudaFree(allChange_device);
	cudaFree(SSSP);
	return 0;
}