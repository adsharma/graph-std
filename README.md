# Graph Standard Formats

This project provides tools to export the Karate Club graph from NetworkX to various standard graph formats.

## Karate Club Graph

The Karate Club graph is a social network of friendships between 34 members of a karate club at a US university in the 1970s. It's a popular dataset in network analysis and graph theory.

## Usage

To export the Karate Club graph to different formats, run:

```bash
python karate.py
```

This will generate five outputs:
- `karate_nodes.csv` and `karate_edges.csv` - CSV format
- `karate.duckdb` - DuckDB database with nodes and edges tables
- `karate.snap` - SNAP format (edge list)
- `karate.snap.bin` - SNAP binary format (efficient binary format)
- `karate_kuzu` - KuzuDB database with nodes and edges tables

## Format Details

### CSV
Two CSV files are generated:
- `karate_nodes.csv`: Contains node IDs and their club affiliation
- `karate_edges.csv`: Contains source and target node IDs for each edge

### DuckDB
A DuckDB database with two tables:
- `nodes`: Contains node_id and club columns
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
- `nodes`: Contains node_id (INT64) and club (STRING) columns with node_id as primary key
- `edges`: A relationship table connecting nodes to nodes