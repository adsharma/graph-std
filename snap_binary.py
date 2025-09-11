import struct
import os
from typing import List, Tuple, Union, Dict, Any


class SNAPBinaryExporter:
    """
    Export graphs to SNAP's efficient binary format.

    SNAP binary format stores:
    - Header with graph metadata
    - Node information
    - Edge list in binary format
    """

    def __init__(self):
        self.version = 1

    def export_graph(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        filename: str,
        directed: bool = True,
        node_attributes: Dict[int, Dict[str, Any]] = None,
        edge_attributes: List[Dict[str, Any]] = None,
    ) -> None:
        """
        Export graph to SNAP binary format.

        Args:
            nodes: List of node IDs
            edges: List of (source, target) tuples
            filename: Output filename
            directed: Whether the graph is directed
            node_attributes: Optional node attributes {node_id: {attr_name: value}}
            edge_attributes: Optional edge attributes [{attr_name: value}] in same order as edges
        """

        with open(filename, "wb") as f:
            # Write header
            self._write_header(
                f,
                len(nodes),
                len(edges),
                directed,
                node_attributes is not None,
                edge_attributes is not None,
            )

            # Write nodes
            self._write_nodes(f, nodes, node_attributes)

            # Write edges
            self._write_edges(f, edges, edge_attributes)

    def _write_header(
        self,
        f,
        num_nodes: int,
        num_edges: int,
        directed: bool,
        has_node_attrs: bool,
        has_edge_attrs: bool,
    ) -> None:
        """Write binary header with graph metadata."""

        # Magic number for identification
        f.write(b"SNAP")

        # Version
        f.write(struct.pack("<I", self.version))

        # Graph properties
        f.write(struct.pack("<I", num_nodes))
        f.write(struct.pack("<I", num_edges))

        # Flags
        flags = 0
        if directed:
            flags |= 1
        if has_node_attrs:
            flags |= 2
        if has_edge_attrs:
            flags |= 4
        f.write(struct.pack("<I", flags))

    def _write_nodes(self, f, nodes: List[int], node_attributes: Dict = None) -> None:
        """Write node information."""

        # Sort nodes for consistent ordering
        sorted_nodes = sorted(nodes)

        # Write node IDs
        for node_id in sorted_nodes:
            f.write(struct.pack("<I", node_id))

        # Write node attributes if present
        if node_attributes:
            self._write_node_attributes(f, sorted_nodes, node_attributes)

    def _write_edges(
        self, f, edges: List[Tuple[int, int]], edge_attributes: List = None
    ) -> None:
        """Write edge list in binary format."""

        # Sort edges for better compression and access patterns
        sorted_edges = sorted(edges)

        # Write edges as pairs of 32-bit integers
        for src, dst in sorted_edges:
            f.write(struct.pack("<II", src, dst))

        # Write edge attributes if present
        if edge_attributes:
            self._write_edge_attributes(f, edge_attributes)

    def _write_node_attributes(self, f, nodes: List[int], attributes: Dict) -> None:
        """Write node attributes section."""

        # Get attribute names
        attr_names = set()
        for node_attrs in attributes.values():
            attr_names.update(node_attrs.keys())
        attr_names = sorted(attr_names)

        # Write number of attributes
        f.write(struct.pack("<I", len(attr_names)))

        # Write attribute names
        for attr_name in attr_names:
            name_bytes = attr_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

        # Write attribute values for each node
        for node_id in nodes:
            node_attrs = attributes.get(node_id, {})
            for attr_name in attr_names:
                value = node_attrs.get(attr_name, 0)  # Default to 0
                if isinstance(value, (int, bool)):
                    f.write(struct.pack("<i", int(value)))
                elif isinstance(value, float):
                    f.write(struct.pack("<f", value))
                else:  # String
                    str_bytes = str(value).encode("utf-8")
                    f.write(struct.pack("<I", len(str_bytes)))
                    f.write(str_bytes)

    def _write_edge_attributes(self, f, attributes: List[Dict]) -> None:
        """Write edge attributes section."""

        if not attributes:
            return

        # Get attribute names from first edge
        attr_names = sorted(attributes[0].keys()) if attributes else []

        # Write number of attributes
        f.write(struct.pack("<I", len(attr_names)))

        # Write attribute names
        for attr_name in attr_names:
            name_bytes = attr_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

        # Write attribute values for each edge
        for edge_attrs in attributes:
            for attr_name in attr_names:
                value = edge_attrs.get(attr_name, 0)
                if isinstance(value, (int, bool)):
                    f.write(struct.pack("<i", int(value)))
                elif isinstance(value, float):
                    f.write(struct.pack("<f", value))
                else:  # String
                    str_bytes = str(value).encode("utf-8")
                    f.write(struct.pack("<I", len(str_bytes)))
                    f.write(str_bytes)


def export_networkx_to_snap(G, filename: str) -> None:
    """
    Convenience function to export NetworkX graph to SNAP binary format.

    Args:
        G: NetworkX graph object
        filename: Output filename
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function")

    # Extract nodes and edges
    nodes = list(G.nodes())
    edges = list(G.edges())

    # Extract attributes
    node_attributes = {}
    for node, attrs in G.nodes(data=True):
        if attrs:
            node_attributes[node] = attrs

    edge_attributes = []
    for _, _, attrs in G.edges(data=True):
        edge_attributes.append(attrs)

    # Create exporter and export
    exporter = SNAPBinaryExporter()
    exporter.export_graph(
        nodes=nodes,
        edges=edges,
        filename=filename,
        directed=G.is_directed(),
        node_attributes=node_attributes if node_attributes else None,
        edge_attributes=edge_attributes if any(edge_attributes) else None,
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Simple graph
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]

    exporter = SNAPBinaryExporter()
    exporter.export_graph(nodes, edges, "simple_graph.snap")

    # Example 2: Graph with attributes
    node_attrs = {
        0: {"label": "start", "weight": 1.0},
        1: {"label": "middle", "weight": 2.5},
        2: {"label": "end", "weight": 3.0},
    }

    edge_attrs = [
        {"weight": 1.2, "type": "strong"},
        {"weight": 0.8, "type": "weak"},
        {"weight": 2.0, "type": "strong"},
    ]

    small_nodes = [0, 1, 2]
    small_edges = [(0, 1), (1, 2), (2, 0)]

    exporter.export_graph(
        nodes=small_nodes,
        edges=small_edges,
        filename="attributed_graph.snap",
        directed=True,
        node_attributes=node_attrs,
        edge_attributes=edge_attrs,
    )

    print("Graphs exported successfully!")

    # Example 3: Using with NetworkX (if available)
    try:
        import networkx as nx

        # Create a sample NetworkX graph
        G = nx.karate_club_graph()
        export_networkx_to_snap(G, "karate_club.snap")
        print("NetworkX graph exported successfully!")

    except ImportError:
        print("NetworkX not available - skipping NetworkX example")
