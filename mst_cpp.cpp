#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <climits>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <thread>
#include <mutex>

#define MAX_NODES 5000
using namespace std;

struct Edge {
  int s; // source
  int d; // destination
  int w; // weight

  // Comparision for sorting to sort in manner of non-decreasing order
  bool operator<(const Edge &second) const {
    return w < second.w;
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
  return parent[i] = find(parent[i], parent);
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

// ============================================================
// Kruskal's Algorithm (Sequential)
// Returns: (mst_weight, elapsed_seconds)
// ============================================================
pair<long long, double> run_kruskal(const vector<Edge> &edges, int n) {
  auto t_start = chrono::high_resolution_clock::now();

  vector<int> parent(n);
  vector<int> rank(n, 0);
  for (int i = 0; i < n; i++) parent[i] = i;

  // Copy edges for sorting (preserves original for other algorithms)
  vector<Edge> edges_copy = edges;

  sort(edges_copy.begin(), edges_copy.end());

  vector<Edge> mst;
  int edges_num = 0;

  for (size_t i = 0; i < edges_copy.size(); i++) {
    const Edge &edge = edges_copy[i];
    if (unionfunction(edge.s, edge.d, parent, rank)) {
      mst.push_back(edge);
      edges_num++;
    }
  }
  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_s = chrono::duration<double>(t_end - t_start).count();

  long long mst_weight = 0;
  for (const auto &e : mst) mst_weight += e.w;

  return {mst_weight, elapsed_s};
}

// ============================================================
// Borůvka's Algorithm (Sequential, No Contraction)
// Algorithm code kept exactly as provided — no modifications
// Returns: (mst_weight, elapsed_seconds)
// ============================================================
pair<long long, double> run_boruvka_seq(const vector<Edge> &edges, int n) {
  auto t_start = chrono::high_resolution_clock::now();

  // parent and rank prepartions
  vector<int> parent(n);
  vector<int> rank(n, 0);

  // Initilize the parent vector
  for (int i = 0; i < n; i++) {
    parent[i] = i; // set each parent with the expected id
  }

  // Step 1: Initilize number of components initially the number of the components will be the nodes maximum number
  int components_num = n;

  // Step 2: start minimum spanning tree
  vector<Edge> mst;
  int edges_num = 0;

  while(components_num > 1){
      vector<int> cheapest(n, -1); // -1 means have not found cheapest yet

      //Part 1: Finding minimum-weight edge
      for(int i = 0; i < (int)edges.size(); i++){
          const Edge &edge = edges[i]; //to get the current edge
          int group1 = find(edge.s, parent); //for the source
          int group2 = find(edge.d, parent); //for the destination

          if(group1 != group2){
              if(cheapest[group1] == -1 || edge.w < edges[cheapest[group1]].w){ //current edge.w with previous edges[cheapest[group1]].w to update the lowest value
                  cheapest[group1] = i;
              }else if(cheapest[group2] == -1 || edge.w < edges[cheapest[group2]].w){ //current edge.w with previous edges[cheapest[group1]].w to update the lowest value
                  cheapest[group2] = i;
              }
          }
      }

      //Part 2: connects different components
      bool merged_any = false;
      for(int i = 0; i < n; i++){
          if(cheapest[i] != -1){
              const Edge &edge = edges[cheapest[i]]; //get current edge

              //push to minimum spanning tree if they are uniqe
              if(unionfunction(edge.s, edge.d, parent, rank)){
                  mst.push_back(edge);
                  components_num--; //because we mereged two diffrenet components
                  merged_any = true;
              }

          }
      }
      if (!merged_any) break; // no progress — graph may be disconnected
  }

  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_s = chrono::duration<double>(t_end - t_start).count();

  long long mst_weight = 0;
  for (const auto &e : mst) mst_weight += e.w;

  return {mst_weight, elapsed_s};
}

// ============================================================
// Borůvka's Algorithm (Parallel, No Contraction)
// Algorithm code kept exactly as provided — no modifications
// Uses std::thread with mutex-guarded per-component cheapest updates
// Returns: (mst_weight, elapsed_seconds)
// ============================================================
pair<long long, double> run_boruvka_par(const vector<Edge> &edges, int n, int num_threads = 0) {
  auto t_start = chrono::high_resolution_clock::now();

  // parent and rank prepartions
  vector<int> parent(n);
  vector<int> rank(n, 0);

  // Initilize the parent vector
  for (int i = 0; i < n; i++) {
    parent[i] = i; // set each parent with the expected id
  }

  // Step 1: Initilize number of components initially the number of the components will be the nodes maximum number
  int components_num = n;

  // Step 2: start minimum spanning tree
  vector<Edge> mst;
  int edges_num = 0;

  // PARALLISIM PREPARTION
  int par_num_threads = (num_threads > 0) ? num_threads : thread::hardware_concurrency();
  int maximum_edges = edges.size();
  int threads_edge = maximum_edges / par_num_threads;
  vector<mutex> component_locks(n);

  while (components_num > 1) {
      vector<int> cheapest(n, -1); // -1 means have not found cheapest yet
      vector<thread> threads;

      // Part 1: Finding minimum-weight edge (PARALLEL WORK LOAD DISTRIBUTION)
      for (int t = 0; t < par_num_threads; t++) {
          //Setup start and end indexes
          int start_index_of_thread = t * threads_edge;
          int end_index_of_thread = 0;

          if(t == par_num_threads - 1){
              end_index_of_thread = maximum_edges;
          }else{
              end_index_of_thread = start_index_of_thread + threads_edge;
          }

          threads.push_back(thread([&, start_index_of_thread, end_index_of_thread]()
          {
              for (int i = start_index_of_thread; i < end_index_of_thread; i++)
              {
                  const Edge &edge = edges[i];       // to get the current edge
                  int group1 = find(edge.s, parent); // for the source
                  int group2 = find(edge.d, parent); // for the destination

                  if (group1 != group2)
                  {
                      {
                          lock_guard<mutex> lock(component_locks[group1]);
                          if (cheapest[group1] == -1 || edge.w < edges[cheapest[group1]].w)
                          {
                              cheapest[group1] = i;
                          }
                      }
                      {
                          lock_guard<mutex> lock(component_locks[group2]);
                          if (cheapest[group2] == -1 || edge.w < edges[cheapest[group2]].w)
                          {
                              cheapest[group2] = i;
                          }
                      }
                  }
              }
          }));
      }

      //join threads
      for(auto &th: threads){
          if(th.joinable()){
              th.join();
          }
      }

      // Part 2: connects different components
      bool merged_any = false;
      for (int i = 0; i < n; i++) {
          if (cheapest[i] != -1) {
              const Edge &edge = edges[cheapest[i]]; // get current edge

              // push to minimum spanning tree if they are uniqe
              if (unionfunction(edge.s, edge.d, parent, rank)) {
                  mst.push_back(edge);
                  components_num--; // because we mereged two diffrenet components
                  merged_any = true;
              }
          }
      }
      if (!merged_any) break; // no progress — graph may be disconnected
  }

  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_s = chrono::duration<double>(t_end - t_start).count();

  long long mst_weight = 0;
  for (const auto &e : mst) mst_weight += e.w;

  return {mst_weight, elapsed_s};
}

// ============================================================
// Main — CLI parsing, graph loading, benchmarking loop
// ============================================================
int main(int argc, char *argv[]) {
  string dataset = "";
  string sizes_raw = "";
  string algorithms_raw = "";
  int num_threads = 1;
  string experiment = "";
  int runs = 1;
  string output_dir = "";

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
  while ((c = getopt_long(argc, argv, "d:s:a:t:e:r:o:", long_options,
                          &option_index)) != -1) {
    switch (c) {
    case 'd': dataset = optarg; break;
    case 's': sizes_raw = optarg; break;
    case 'a': algorithms_raw = optarg; break;
    case 't': num_threads = stoi(optarg); break;
    case 'e': experiment = optarg; break;
    case 'r': runs = stoi(optarg); break;
    case 'o': output_dir = optarg; break;
    default:
      cerr << "Usage error.\n";
      return 1;
    }
  }

  vector<string> sizes_str =
      split(sizes_raw.empty() ? to_string(MAX_NODES) : sizes_raw, ',');

  // Parse algorithms (default: kruskal,boruvka_seq)
  set<string> algorithms;
  {
    vector<string> alg_list = split(
        algorithms_raw.empty() ? "kruskal,boruvka_seq" : algorithms_raw, ',');
    for (const auto &a : alg_list) algorithms.insert(a);
  }
  bool do_kruskal = algorithms.count("kruskal") > 0;
  bool do_boruvka = algorithms.count("boruvka_seq") > 0;
  bool do_boruvka_par = algorithms.count("boruvka_par") > 0;

  string ds_name = filesystem::path(
      dataset.empty() ? "Amazon0302.txt" : dataset).stem().string();

  string csv_path;
  bool csv_needs_header = false;
  if (!output_dir.empty()) {
    filesystem::create_directories(output_dir);
    csv_path = output_dir + "/scalability_" + ds_name + ".csv";
    csv_needs_header = !filesystem::exists(csv_path);
  }

  for (const string &size_str : sizes_str) {
    int active_max_nodes = stoi(size_str);
    bool is_full = (active_max_nodes == 0);
    if (is_full) active_max_nodes = INT_MAX;

    // ── Load graph (once per size) ──
    minstd_rand lcg(42);
    map<pair<int, int>, int> edge_map;
    ifstream file(dataset.empty() ? "Amazon0302.txt" : dataset);
    string line;
    while (getline(file, line)) {
      if (line.empty() || line[0] == '#') continue;
      stringstream ss(line);
      int sn, dn;
      if (ss >> sn >> dn) {
        if (sn < active_max_nodes && dn < active_max_nodes) {
          int s = min(sn, dn);
          int d = max(sn, dn);
          auto key = make_pair(s, d);
          if (edge_map.find(key) == edge_map.end()) {
            edge_map[key] = int(lcg());
          }
        }
      }
    }
    file.close();

    vector<Edge> edges;
    for (auto itr = edge_map.begin(); itr != edge_map.end(); ++itr) {
      edges.push_back({itr->first.first, itr->first.second, itr->second});
    }

    // Count vertices two ways:
    // - num_nodes: max_id + 1, needed for parent/rank array sizing in algorithms
    // - num_unique_vertices: actual distinct vertex count, for CSV (matches Python/Rust)
    int num_nodes = 0;
    set<int> unique_vertices;
    for (const auto &e : edges) {
      num_nodes = max(num_nodes, max(e.s, e.d) + 1);
      unique_vertices.insert(e.s);
      unique_vertices.insert(e.d);
    }
    int num_unique_vertices = unique_vertices.size();

    // Helper lambda to benchmark an algorithm and write CSV
    auto bench_algo = [&](const string &algo_name,
                          function<pair<long long, double>(const vector<Edge>&, int)> algo_fn) {
      vector<double> times;
      long long final_mst_weight = 0;

      for (int run = 1; run <= runs; ++run) {
        auto [mst_weight, elapsed_s] = algo_fn(edges, num_nodes);
        times.push_back(elapsed_s);
        final_mst_weight = mst_weight;

        cout << "  V=" << num_unique_vertices << ", E=" << edge_map.size()
             << "  " << algo_name << ": " << elapsed_s << "s"
             << "  (MST weight=" << mst_weight << ")" << endl;
      }

      double median_s = compute_median(times);
      double mean_s = compute_mean(times);
      double std_s = compute_std(times, mean_s);
      double min_s = *min_element(times.begin(), times.end());
      double max_s = *max_element(times.begin(), times.end());

      if (!output_dir.empty()) {
        ofstream csv(csv_path, ios::app);
        if (csv_needs_header) {
          csv << "dataset,algorithm,n_vertices,n_edges,threads,run,time_s,mst_weight,"
              << "median_s,mean_s,std_s,min_s,max_s" << endl;
          csv_needs_header = false;
        }
        int csv_threads = (algo_name == "boruvka_par") ? ((num_threads > 0) ? num_threads : (int)thread::hardware_concurrency()) : 1;
        for (int run = 0; run < runs; run++) {
          csv << ds_name << "," << algo_name << "," << num_unique_vertices << ","
              << edge_map.size() << "," << csv_threads << "," << run << "," << times[run] << ","
              << final_mst_weight << "," << median_s << "," << mean_s << ","
              << std_s << "," << min_s << "," << max_s << endl;
        }
        csv.close();
      }
    };

    if (do_kruskal)     bench_algo("kruskal", run_kruskal);
    if (do_boruvka)      bench_algo("boruvka_seq", run_boruvka_seq);
    if (do_boruvka_par)  bench_algo("boruvka_par", [&](const vector<Edge>& e, int nn) {
      return run_boruvka_par(e, nn, num_threads);
    });
  }
  return 0;
}