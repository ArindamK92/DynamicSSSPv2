#ifndef ALL_STRUCTURE_CUH
#define ALL_STRUCTURE_CUH

#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
using namespace std;

/******* Network Structures *********/
struct ColWt {
	int col;
	int wt;
};

//Structure for Edge
struct Edge
{
	int node1;
	int node2;
	double edge_wt;
};


struct changeEdge {
	int node1;
	int node2;
	double edge_wt;
	int inst;
};

typedef vector<ColWt> ColWtList;





// Data Structure for each vertex in the rooted tree
struct RT_Vertex
{
	//int Root;    //root fo the tree
	int Parent; //mark the parent in the tree
	int EDGwt; //mark weight of the edge
	//int marked; //whether the vertex and the edge connecting its parent ..
				//..exists(-1); has been deleted(-2); is marked for replacement (+ve value index of changed edge)

	int Dist;  //Distance from root
	bool Update;  //Whether the distance of this edge was updated / affected
};


////functions////
//Node starts from 0
//reads only the edges in the edge list, does not reverse them to make undirected



/*
 readin_changes function reads the change edges
 Format of change edges file: node1 node2 edge_weight insert_status
 insert_status = 1 for insertion. insert_status = 0 for deletion.
 */
void readin_changes(char* myfile, vector<changeEdge>& allChange, vector<ColWtList>& InEdgesList, vector<ColWtList>& OutEdgesList,int& totalInsertion)
{
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt, inst_status;
		changeEdge cE;
		sscanf(line, "%d %d %d %d", &n1, &n2, &wt, &inst_status);
		cE.node1 = n1;
		cE.node2 = n2;
		cE.edge_wt = wt;
		cE.inst = inst_status;
		allChange.push_back(cE);

		//add change edges with inst status = 1 to Inedge and OutEdge list
		if (inst_status == 1)
		{
			totalInsertion++;
			ColWt colwt;
			colwt.col = n1;
			colwt.wt = wt;
			InEdgesList.at(n2).push_back(colwt);
			ColWt colwt2;
			colwt2.col = n2;
			colwt2.wt = wt;
			OutEdgesList.at(n1).push_back(colwt2);
		}
	}
	fclose(graph_file);
	return;
}


/*
read_SSSP reads input SSSP file.
accepted SSSP data format: node parent distance
*/
void read_SSSP(RT_Vertex* SSSP, char* myfile, int* nodes)
{
	FILE* graph_file;
	char line[128];
	
	int prev_node = 0;
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int node, parent, dist;
		sscanf(line, "%d %d %d", &node, &parent, &dist);
		if (node > prev_node + 1){
			for (int i = prev_node + 1; i < node; i++)
			{
				SSSP[i].Dist = 9999999;
				SSSP[i].Parent = -1;
				//SSSP[i].EDGwt = 9999999;
			}
		}
		if (parent == -1)
		{
			dist = 9999999;
		}
		SSSP[node].Parent = parent;
		SSSP[node].Dist = dist;
		prev_node = node;
	}
	for (int j = prev_node + 1; j < *nodes; j++)
	{
		SSSP[j].Dist = 9999999;
		SSSP[j].Parent = -1;
		//SSSP[i].EDGwt = 9999999;
	}
	fclose(graph_file);

	return;
}

/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
*/
void read_graphEdges(vector<ColWtList>& InEdgesList, /*int* InEdgesListTracker, vector<ColWt>& InEdgesListFull,*/ char* myfile, int* nodes, vector<ColWtList>& OutEdgesList/*, int* OutEdgesListTracker, vector<ColWt>& OutEdgesListFull*/)
{
	FILE* graph_file;
	char line[128];


	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt;
		sscanf(line, "%d %d %d", &n1, &n2, &wt);
		ColWt colwt;
		colwt.col = n1;
		colwt.wt = wt;
		InEdgesList.at(n2).push_back(colwt);
		ColWt colwt2;
		colwt2.col = n2;
		colwt2.wt = wt;
		OutEdgesList.at(n1).push_back(colwt2);

	}
	fclose(graph_file);
	//InEdgesListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	//OutEdgesListTracker[0] = 0; //start pointer points to the first index of OutEdgesList
	//for (int i = 0; i < *nodes; i++)
	//{
	//	InEdgesListTracker[i + 1] = InEdgesListTracker[i] + InEdgesList.at(i).size();
	//	InEdgesListFull.insert(std::end(InEdgesListFull), std::begin(InEdgesList.at(i)), std::end(InEdgesList.at(i)));
	//	OutEdgesListTracker[i + 1] = OutEdgesListTracker[i] + OutEdgesList.at(i).size();
	//	OutEdgesListFull.insert(std::end(OutEdgesListFull), std::begin(OutEdgesList.at(i)), std::end(OutEdgesList.at(i)));
	//}

	return;
}

#endif