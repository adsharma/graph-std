# Graph Standard Formats

This project provides tools to export various graphs from NetworkX to various standard graph formats, including CSR format for efficient processing.

## Supported Graphs

- Karate Club graph: A social network of friendships between 34 members of a karate club
- Complete graph: A graph where every pair of distinct vertices is connected by a unique edge
- Cycle graph: A graph that consists of a single cycle
- Path graph: A graph whose vertices can be listed in an order such that the edges connect consecutive vertices
- Kronecker graph: A scale-free graph with properties similar to Kronecker graphs

## Usage

To generate graph data in various formats, run:

```bash
python gen.py [--type TYPE] [--size SIZE] [--randomize-ids]
```

Options:
- `--type`: Type of graph to generate (karate, complete, cycle, path, kronecker) - default: karate
- `--size`: Size of the graph (number of nodes for applicable graph types)
- `--randomize-ids`: Randomize node IDs in a space 10x the number of nodes

This will generate five outputs:
- `{prefix}_nodes.csv` and `{prefix}_edges.csv` - CSV format
- `{prefix}.duckdb` - DuckDB database with nodes and edges tables
- `{prefix}.snap` - SNAP format (edge list)
- `{prefix}.snap.bin` - SNAP binary format (efficient binary format)
- `{prefix}_kuzu` - KuzuDB database with nodes and edges tables

Example for Karate Club graph:
```bash
python gen.py --type karate
```

Example for a randomized Karate Club graph:
```bash
python gen.py --type karate --randomize-ids
```

Example for a complete graph with 20 nodes:
```bash
python gen.py --type complete --size 20
```

## Converting to CSR Format

To convert a DuckDB graph to Compressed Sparse Row (CSR) format for efficient processing:

```bash
python convert_csr.py --source-db SOURCE_DB --output-db OUTPUT_DB [--csr-table TABLE_NAME] [--directed] [--test --limit LIMIT]
```

Options:
- `--source-db`: Source DuckDB database path (default: karate_random.duckdb)
- `--output-db`: Output DuckDB database path (default: csr_graph.db)
- `--csr-table`: Table name prefix for CSR data (default: csr_graph)
- `--directed`: Treat graph as directed (default: undirected)
- `--test`: Run in test mode with limited data
- `--limit`: Number of edges to use in test mode (default: 50000)

Example:
```bash
python convert_csr.py --source-db karate_random.duckdb --csr-table karate_random --output-db karate_csr.db
```

This will create a CSR representation with four tables:
- `{table_name}_node_mapping`: Maps original node IDs to contiguous indices
- `{table_name}_indptr`: Array of size N+1 for row pointers
- `{table_name}_indices`: Array of size E containing column indices
- `{table_name}_metadata`: Graph metadata (node count, edge count, directed flag)

## Format Details

### CSV
Two CSV files are generated:
- `{prefix}_nodes.csv`: Contains node IDs and their attributes (if any)
- `{prefix}_edges.csv`: Contains source and target node IDs for each edge

### DuckDB
A DuckDB database with two tables:
- `nodes`: Contains node_id and attributes (if any)
- `edges`: Contains source and target columns

### SNAP
A text file with the SNAP format:
- Comment lines starting with #
- Each line represents an edge with source and target node IDs

### SNAP Binary
A binary file with the SNAP binary format:
- Efficient binary representation of graph data
- Contains header with graph metadata
- Stores nodes and edges in a compact binary format
- Supports optional node and edge attributes

### KuzuDB
A KuzuDB database with two tables:
- `nodes`: Contains node_id (INT64) and attributes (if any) with node_id as primary key
- `edges`: A relationship table connecting nodes to nodes

### CSR (Compressed Sparse Row)
A representation optimized for fast graph algorithms:
- Node mapping: Maps original sparse node IDs to 0-based contiguous indices
- Row pointers (indptr): For each node, points to the start of its edges in the indices array
- Column indices (indices): Contains the target node for each edge
- Metadata: Stores graph properties (node count, edge count, directed flag)