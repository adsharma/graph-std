#!/usr/bin/env python3
"""
Script to dump the Karate Club graph from NetworkX to various formats:
- CSV (nodes and edges separately)
- DuckDB (nodes and edges in separate tables)
- SNAP (simple edge list format)
- SNAP Binary (efficient binary format)
- KuzuDB (nodes and edges in separate tables)

This script also supports randomizing node IDs in a space that's 10x the size of
the number of nodes in the graph.
"""

import argparse
import networkx as nx
import csv
import duckdb
import kuzu
import pandas as pd
import random
from snap_binary import export_networkx_to_snap


def load_karate_graph():
    """Load the Karate Club graph from NetworkX."""
    return nx.karate_club_graph()


def load_graph(graph_type, size=None):
    """Load a graph based on the specified type and optional size."""
    if graph_type == "karate":
        return load_karate_graph()
    elif graph_type == "complete":
        if size is None:
            size = 10  # Default size for complete graph
        return nx.complete_graph(size)
    elif graph_type == "cycle":
        if size is None:
            size = 10  # Default size for cycle graph
        return nx.cycle_graph(size)
    elif graph_type == "path":
        if size is None:
            size = 10  # Default size for path graph
        return nx.path_graph(size)
    elif graph_type == "kronecker":
        # Using scale-free graph as an approximation to Kronecker graphs
        if size is None:
            size = 10  # Default size for kronecker graph
        # Generate a scale-free graph with properties similar to Kronecker graphs
        return nx.scale_free_graph(size, seed=42)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")


def randomize_node_ids(graph):
    """
    Randomize node IDs in a space that's 10x the size of the number of nodes.

    Args:
        graph: NetworkX graph

    Returns:
        NetworkX graph with randomized node IDs
    """
    num_nodes = graph.number_of_nodes()
    max_id_space = num_nodes * 10

    # Generate unique random IDs for each node
    original_nodes = list(graph.nodes())
    random_ids = random.sample(range(max_id_space), num_nodes)

    # Create mapping from original IDs to randomized IDs
    id_mapping = dict(zip(original_nodes, random_ids))

    # Create new graph with randomized IDs
    if hasattr(graph, "to_directed"):
        new_graph = graph.__class__()
    else:
        new_graph = nx.Graph()

    # Add nodes with new IDs
    for node in graph.nodes():
        new_graph.add_node(id_mapping[node], **graph.nodes[node])

    # Add edges with new IDs
    for edge in graph.edges():
        new_graph.add_edge(id_mapping[edge[0]], id_mapping[edge[1]])

    return new_graph


def export_to_csv(graph, prefix="karate"):
    """Export the graph to CSV files (nodes and edges)."""
    # Export nodes
    with open(f"{prefix}_nodes.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Check if the graph has club attribute
        has_club = any("club" in data for node, data in graph.nodes(data=True))
        if has_club:
            writer.writerow(["id", "club"])
            for node, data in graph.nodes(data=True):
                writer.writerow([node, data.get("club", "")])
        else:
            writer.writerow(["id"])
            for node in graph.nodes():
                writer.writerow([node])

    # Export edges
    with open(f"{prefix}_edges.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source", "target"])
        for edge in graph.edges():
            writer.writerow(edge)

    print(f"Graph exported to {prefix}_nodes.csv and {prefix}_edges.csv")


def export_to_duckdb(graph, db_name="karate.duckdb"):
    """Export the graph to a DuckDB database."""
    # Connect to DuckDB
    conn = duckdb.connect(db_name)

    # Check if the graph has club attribute
    has_club = any("club" in data for node, data in graph.nodes(data=True))

    # Create nodes dataframe
    if has_club:
        nodes_data = [
            (node, data.get("club", "")) for node, data in graph.nodes(data=True)
        ]
        nodes_df = pd.DataFrame(nodes_data, columns=["id", "club"])
    else:
        nodes_data = [(node,) for node in graph.nodes()]
        nodes_df = pd.DataFrame(nodes_data, columns=["id"])

    # Create edges dataframe
    edges_data = [(edge[0], edge[1]) for edge in graph.edges()]
    edges_df = pd.DataFrame(edges_data, columns=["source", "target"])

    # Drop tables if they exist
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("DROP TABLE IF EXISTS edges")

    # Create tables and insert data using dataframes (bulk insert)
    # Using DuckDB's native support for dataframes
    conn.execute("CREATE TABLE nodes AS SELECT * FROM nodes_df")
    conn.execute("CREATE TABLE edges AS SELECT * FROM edges_df")

    # Close connection
    conn.close()

    print(f"Graph exported to {db_name}")


def export_to_snap(graph, filename="karate.snap"):
    """Export the graph to SNAP format (edge list)."""
    with open(filename, "w") as f:
        # Write header comments
        f.write(f"# {filename.split('.')[0]} graph\n")
        f.write(
            f"# Nodes: {graph.number_of_nodes()} Edges: {graph.number_of_edges()}\n"
        )

        # Write edges
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Graph exported to {filename}")


def export_to_snap_binary(graph, filename="karate.snap.bin"):
    """Export the graph to SNAP binary format."""
    export_networkx_to_snap(graph, filename)
    print(f"Graph exported to {filename}")


def export_to_kuzudb(graph, db_name="karate_kuzu"):
    """Export the graph to a KuzuDB database."""
    # Create or open KuzuDB database
    db = kuzu.Database(db_name)
    conn = kuzu.Connection(db)

    # Check if the graph has club attribute
    has_club = any("club" in data for node, data in graph.nodes(data=True))

    # Drop tables if they exist
    try:
        conn.execute("DROP TABLE edges")
    except RuntimeError:
        # Table might not exist, which is fine
        pass

    try:
        conn.execute("DROP TABLE nodes")
    except RuntimeError:
        # Table might not exist, which is fine
        pass

    # Create nodes table
    if has_club:
        conn.execute(
            "CREATE NODE TABLE nodes(id INT64, club STRING, PRIMARY KEY (id))"
        )

        # Create DataFrame for nodes
        nodes_data = [
            (node, data.get("club", "")) for node, data in graph.nodes(data=True)
        ]
        nodes_df = pd.DataFrame(nodes_data, columns=["id", "club"])

        # Use COPY to efficiently load nodes from DataFrame
        conn.execute("COPY nodes FROM nodes_df")
    else:
        conn.execute("CREATE NODE TABLE nodes(id INT64, PRIMARY KEY (id))")

        # Create DataFrame for nodes
        nodes_data = [(node,) for node in graph.nodes()]
        nodes_df = pd.DataFrame(nodes_data, columns=["id"])

        # Use COPY to efficiently load nodes from DataFrame
        conn.execute("COPY nodes FROM nodes_df")

    # Create edges table
    conn.execute("CREATE REL TABLE edges(FROM nodes TO nodes)")

    # Create DataFrame for edges
    edges_data = [(edge[0], edge[1]) for edge in graph.edges()]
    edges_df = pd.DataFrame(edges_data, columns=["source", "target"])

    # Use COPY to efficiently load edges from DataFrame
    conn.execute("COPY edges FROM edges_df")

    print(f"Graph exported to KuzuDB database: {db_name}")


def main():
    """Main function to load graph and export to all formats."""
    parser = argparse.ArgumentParser(
        description="Generate graph data in various formats"
    )
    parser.add_argument(
        "--type",
        choices=["karate", "complete", "cycle", "path", "kronecker"],
        default="karate",
        help="Type of graph to generate (default: karate)",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Size of the graph (number of nodes for applicable graph types)",
    )
    parser.add_argument(
        "--randomize-ids",
        action="store_true",
        help="Randomize node IDs in a space 10x the number of nodes",
    )

    args = parser.parse_args()

    print(f"Loading {args.type} graph...")
    try:
        graph = load_graph(args.type, args.size)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(
        f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # Randomize node IDs if requested
    if args.randomize_ids:
        print("Randomizing node IDs...")
        graph = randomize_node_ids(graph)
        print(
            f"Node IDs randomized: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

    # Generate prefix for output files
    prefix = args.type
    if args.size:
        prefix += f"_{args.size}"
    if args.randomize_ids:
        prefix += "_random"

    print("Exporting to CSV...")
    export_to_csv(graph, prefix)

    print("Exporting to DuckDB...")
    export_to_duckdb(graph, f"{prefix}.duckdb")

    print("Exporting to SNAP...")
    export_to_snap(graph, f"{prefix}.snap")

    print("Exporting to SNAP Binary...")
    export_to_snap_binary(graph, f"{prefix}.snap.bin")

    print("Exporting to KuzuDB...")
    export_to_kuzudb(graph, f"{prefix}_kuzu")

    print("All exports completed!")


if __name__ == "__main__":
    main()
