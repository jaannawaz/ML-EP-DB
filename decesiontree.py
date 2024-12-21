from sklearn.tree import export_graphviz
from io import StringIO  # Use Python's io module
import pydotplus
import joblib
from IPython.display import Image

# Load the trained model
model_file = "trained_model.pkl"
model = joblib.load(model_file)

# Function to visualize and save a specific part of the decision tree
def save_tree_section(model, feature_names, class_names, start_depth, end_depth, file_name):
    dot_data = StringIO()
    export_graphviz(
        model.estimators_[0],  # Accessing one tree from the RandomForest model for visualization
        out_file=dot_data,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,  # Rounded corners
        proportion=False,
        precision=2,
        filled=True,
        node_ids=True,
        max_depth=end_depth
    )

    # Use pydotplus to generate a graph from the DOT data
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # Modify the shapes to circles instead of rectangles
    for node in graph.get_node_list():
        if node.get_shape() == "box":
            node.set_shape("circle")

    # Save the graph in a modern format
    graph_file_path = f"save/{file_name}.png"
    graph.write_png(graph_file_path)

    # Display the saved decision tree graph
    display_image = Image(graph_file_path)
    return display_image

# Define feature names and class names
feature_names = ["No_Pathways_normalized", "CADD_Score_normalized", "Log_FC_normalized", "MGI_Phenotpe_normalized", "Phen2Gene_Score_normalized"]
class_names = ["Non-EP Associated", "EP Associated"]

# Save the top part of the decision tree
print("Saving top part of the decision tree...")
top_image = save_tree_section(model, feature_names, class_names, start_depth=0, end_depth=2, file_name="tree_top")
top_image

# Save the left branch part of the decision tree
print("Saving left branch part of the decision tree...")
left_image = save_tree_section(model, feature_names, class_names, start_depth=2, end_depth=5, file_name="tree_left_branch")
left_image

# Save the right branch part of the decision tree
print("Saving right branch part of the decision tree...")
right_image = save_tree_section(model, feature_names, class_names, start_depth=5, end_depth=8, file_name="tree_right_branch")
right_image

print("All sections of the decision tree have been saved.")
