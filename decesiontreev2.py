import pandas as pd
import networkx as nx

# Load the dataset
input_file = "Final-Normalized-data-EP-Ass-NonAss.csv"
df = pd.read_csv(input_file)

# Prepare the data for Cytoscape
nodes = []
edges = []

# Extract nodes and edges from decision tree (sample structure)
def extract_tree_nodes_and_edges():
    # Sample node and edge structure for Cytoscape import
    nodes.append({'node_id': 0, 'label': 'Root Node'})
    nodes.append({'node_id': 1, 'label': 'Node 1'})
    nodes.append({'node_id': 2, 'label': 'Node 2'})
    edges.append({'source': 0, 'target': 1})
    edges.append({'source': 0, 'target': 2})

extract_tree_nodes_and_edges()

# Create a NetworkX graph
graph = nx.DiGraph()

# Add nodes and edges to the graph
for node in nodes:
    graph.add_node(node['node_id'], label=node['label'])

for edge in edges:
    graph.add_edge(edge['source'], edge['target'])

# Export the graph to a format suitable for Cytoscape
cytoscape_export_file = "decision_tree_network.graphml"
nx.write_graphml(graph, cytoscape_export_file)

print(f"Graph exported successfully to {cytoscape_export_file} for Cytoscape.")
