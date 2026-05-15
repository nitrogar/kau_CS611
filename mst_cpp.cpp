#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

#define MAX_NODES 5000
using namespace std;

struct Edge {
  int s; // source
  int d; // destination
  int w; // weight

  // Comparision for sorting to sort in manner of non-decreasing order
  bool operator<(const Edge &second) const {
    return w < second.w; // w is the current w of current edge vector, second.w
                         // is the weight of the next vector
  }
};

/// @brief used to find tree
/// @param i is the node id
/// @param parent is the parent vector
/// @return modifications in the parent vector
int find(int i, vector<int> &parent) {
  if (parent[i] == i) {
    return i;
  }
  return parent[i] = find(parent[i],
                          parent); // recursive search if parent is not the root
}

/// @brief it is union function used to set unin and check for cyclicity
/// @param sn source node
/// @param dn destination node
/// @param parent parent vector
/// @param rank rank vector
/// @return true if no cycle appears and false if cycle is appearing
bool unionfunction(int sn, int dn, vector<int> &parent, vector<int> &rank) {
  int rootSN = find(sn, parent);
  int rootDN = find(dn, parent);

  if (rootSN != rootDN) {
    if (rank[rootSN] < rank[rootDN]) {
      parent[rootSN] = rootDN;
    } else if (rank[rootSN] > rank[rootDN]) {
      parent[rootDN] = rootSN;
    } else {
      parent[rootSN] = rootDN;
      rank[rootDN]++;
    }
    return true; // no cyclicity
  }

  return false; // cyclicity detected
}

// Helper function to split string arguments like sizes (e.g., "1000,2000,5000")
vector<string> split(const string &s, char delimiter) {
  vector<string> tokens;
  string token;
  istringstream tokenStream(s);
  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// ============================================================
// Stats helpers (for CSV output matching Rust/Python format)
// ============================================================
double compute_median(vector<double> v) {
  size_t n = v.size();
  sort(v.begin(), v.end());
  if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
  return v[n/2];
}

double compute_mean(const vector<double> &v) {
  double sum = 0;
  for (double x : v) sum += x;
  return sum / v.size();
}

double compute_std(const vector<double> &v, double mean) {
  double sum_sq = 0;
  for (double x : v) sum_sq += (x - mean) * (x - mean);
  return sqrt(sum_sq / v.size());
}

int main(int argc, char *argv[]) {
  // CLI Flag Variables
  string dataset = "";
  string sizes_raw = "";
  string algorithms_raw = "";
  int num_threads = 1;
  string experiment = "";
  int runs = 1;
  string output_dir = "";

  // Mapping long flags to short options
  static struct option long_options[] = {
      {"dataset", required_argument, 0, 'd'},
      {"sizes", required_argument, 0, 's'},
      {"algorithms", required_argument, 0, 'a'},
      {"num-threads", required_argument, 0, 't'},
      {"experiment", required_argument, 0, 'e'},
      {"runs", required_argument, 0, 'r'},
      {"output-dir", required_argument, 0, 'o'},
      {0, 0, 0, 0}};

  int option_index = 0;
  int c;

  // Parse the command line arguments
  while ((c = getopt_long(argc, argv, "d:s:a:t:e:r:o:", long_options,
                          &option_index)) != -1) {
    switch (c) {
    case 'd':
      dataset = optarg;
      break;
    case 's':
      sizes_raw = optarg;
      break;
    case 'a':
      algorithms_raw = optarg;
      break;
    case 't':
      num_threads = stoi(optarg);
      break;
    case 'e':
      experiment = optarg;
      break;
    case 'r':
      runs = stoi(optarg);
      break;
    case 'o':
      output_dir = optarg;
      break;
    default:
      cerr << "Usage error.\n";
      return 1;
    }
  }

  // Default to MAX_NODES if sizes flag isn't provided
  vector<string> sizes_str =
      split(sizes_raw.empty() ? to_string(MAX_NODES) : sizes_raw, ',');

  // Extract dataset name from path
  string ds_name = filesystem::path(
      dataset.empty() ? "Amazon0302.txt" : dataset).stem().string();

  // Prepare CSV output
  string csv_path;
  bool csv_needs_header = false;
  if (!output_dir.empty()) {
    filesystem::create_directories(output_dir);
    csv_path = output_dir + "/scalability_" + ds_name + ".csv";
    csv_needs_header = !filesystem::exists(csv_path);
  }

  // Loop through each size provided in the --sizes flag
  for (const string &size_str : sizes_str) {
    int active_max_nodes = stoi(size_str);
    bool is_full = (active_max_nodes == 0);
    // 0 means "load full dataset" — set a very large threshold
    if (is_full) active_max_nodes = INT_MAX;

    // Collect timing data across all runs for this size
    vector<double> times;
    long long final_mst_weight = 0;
    int final_edge_count = 0;
    int actual_nodes = 0;

    // Loop to handle multiple benchmark iterations requested by --runs
    for (int run = 1; run <= runs; ++run) {

      // Step 1:
      // Random data for setting weight for edges
      minstd_rand lcg(42); // the seed is the same in python

      // map to fetch data from text file
      map<pair<int, int>, int> edge_map;

      // load data
      ifstream file(dataset.empty() ? "Amazon0302.txt" : dataset);

      // get text from the file
      string line;
      while (getline(file, line)) {
        // This is to skip the header text and comments
        if (line.empty() || line[0] == '#') {
          continue;
        }

        // After skiping the header parsing data is the next operation
        stringstream ss(line);
        int sn, dn; // sn is FromNodeId in the text file, dn is the ToNodeId in
                    // the text file
        if (ss >> sn >> dn) {
          if (sn < active_max_nodes &&
              dn < active_max_nodes) // To do experements on diffrenet size of
                                     // nodes
          {
            int s = min(sn, dn);
            int d = max(sn, dn);
            pair<int, int> source_destination_pair = make_pair(s, d);

            // Check if the connection is not already there
            if (edge_map.find(source_destination_pair) == edge_map.end()) {
              edge_map[source_destination_pair] = int(lcg());
            }
          }
        }
      }
      file.close();

      vector<Edge> edges; // To store edges

      map<pair<int, int>, int>::iterator itr =
          edge_map.begin(); // create iterator to edge_map map

      for (size_t i = 0; i < edge_map.size(); i++) {
        // SOURCE         DESTINATION        WEIGHT
        edges.push_back({itr->first.first, itr->first.second, itr->second});
        itr++; // increament the iterator
      }

      // parent and rank prepartions
      vector<int> parent(active_max_nodes);
      vector<int> rank(active_max_nodes, 0);

      // Initilize the parent vector
      for (int i = 0; i < active_max_nodes; i++) {
        parent[i] = i; // set each parent with the expected id
      }

      // Step 1: Sorting Edge by w in non-decreasing order
      // Timer starts HERE — covers sort + MST (matches Rust/Python timing)
      auto t_start = chrono::high_resolution_clock::now();
      sort(edges.begin(), edges.end());

      // Step 2: start minimum spanning tree
      vector<Edge> mst;
      int edges_num = 0;

      clock_t start_time = clock();
      for (size_t i = 0; i < edges.size();
           i++) // Changed to size_t to resolve signed compiler warnings
      {
        const Edge &edge = edges[i];

        if (unionfunction(edge.s, edge.d, parent, rank)) {
          mst.push_back(edge);
          edges_num++;
        }
      }
      clock_t end_time = clock();
      auto t_end = chrono::high_resolution_clock::now();
      double elapsed_s = chrono::duration<double>(t_end - t_start).count();

      // Compute MST weight
      long long mst_weight = 0;
      for (const auto &e : mst) mst_weight += e.w;

      // Determine actual vertex count (for full-dataset runs)
      actual_nodes = active_max_nodes;
      if (is_full) {
        actual_nodes = 0;
        for (const auto &e : edges) actual_nodes = max(actual_nodes, max(e.s, e.d) + 1);
      }

      times.push_back(elapsed_s);
      final_mst_weight = mst_weight;
      final_edge_count = (int)edge_map.size();

      cout << "  V=" << actual_nodes << ", E=" << edge_map.size()
           << "  Kruskal (Seq): " << elapsed_s << "s"
           << "  (MST weight=" << mst_weight << ")" << endl;

      if (runs > 1 || sizes_str.size() > 1)
        cout << "------------------------------------" << endl;
    }

    // Compute stats across all runs for this size
    double median_s = compute_median(times);
    double mean_s = compute_mean(times);
    double std_s = compute_std(times, mean_s);
    double min_s = *min_element(times.begin(), times.end());
    double max_s = *max_element(times.begin(), times.end());

    // Write CSV rows (all runs for this size, with shared stats)
    if (!output_dir.empty()) {
      ofstream csv(csv_path, ios::app);
      if (csv_needs_header) {
        csv << "dataset,algorithm,n_vertices,n_edges,threads,run,time_s,mst_weight,"
            << "median_s,mean_s,std_s,min_s,max_s" << endl;
        csv_needs_header = false;
      }
      for (int run = 0; run < runs; run++) {
        csv << ds_name << ",kruskal," << actual_nodes << "," << final_edge_count
            << ",1," << run << "," << times[run] << "," << final_mst_weight
            << "," << median_s << "," << mean_s << "," << std_s
            << "," << min_s << "," << max_s << endl;
      }
      csv.close();
    }
  }
  return 0;
}