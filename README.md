# Graph Standard Formats

This project provides tools to convert graph data from simple DuckDB databases or Parquet files containing `nodes_*` and `edges_*` tables, along with a `schema.cypher` file, into standardized graph formats for efficient processing.

## Converting to CSR Format

To convert a DuckDB graph to Compressed Sparse Row (CSR) format for efficient processing:

```bash
uv run convert_csr.py --source-db SOURCE_DB --output-db OUTPUT_DB --schema path/to/schema.cypher [--csr-table TABLE_NAME] [--directed] [--test --limit LIMIT]
```

Options:
- `--source-db`: Source DuckDB database path (default: karate_random.duckdb)
- `--output-db`: Output DuckDB database path (default: csr_graph.db)
- `--csr-table`: Table name prefix for CSR data (default: csr_graph)
- `--directed`: Treat graph as directed (default: undirected)
- `--test`: Run in test mode with limited data
- `--limit`: Number of edges to use in test mode (default: 50000)
- `--schema`: Path to schema.cypher for edge relationship info (FROM/TO node types)

Example:
```bash
python convert_csr.py --source-db karate_random.duckdb --csr-table karate_random --output-db karate_csr.db --schema schema.cypher
```

This will create a CSR representation with multiple tables depending on the number of node and edge types:

- `{table_name}_indptr_{edge_name}`: Array of size N+1 for row pointers (one per edge table)
- `{table_name}_indices_{edge_name}`: Array of size E containing column indices (one per edge table)
- `{table_name}_nodes_{node_name}`: Original nodes table with node attributes (one per node table)
- `{table_name}_mapping_{node_name}`: Maps original node IDs to contiguous indices (one per node table)
- `{table_name}_metadata`: Global graph metadata (node count, edge count, directed flag)
- `schema.cypher`: A cypher schema that a graph database can mount without ingesting
