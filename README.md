# Graph Standard Format (graph-std)

This project provides tools to convert graph data from simple DuckDB databases or Parquet files containing `nodes_*` and `edges_*` tables, along with a `schema.cypher` file, into standardized graph formats for efficient processing.

## Converting to CSR Format

```bash
uv run python convert_csr.py \
--source-db karate/karate_random.duckdb \
--output-db karate/karate_csr.duckdb \
--csr-table karate \
--schema karate/karate_csr/schema.cypher
```

This will create a CSR representation with multiple tables depending on the number of node and edge types:

- `{table_name}_indptr_{edge_name}`: Array of size N+1 for row pointers (one per edge table)
- `{table_name}_indices_{edge_name}`: Array of size E containing column indices (one per edge table)
- `{table_name}_nodes_{node_name}`: Original nodes table with node attributes (one per node table)
- `{table_name}_mapping_{node_name}`: Maps original node IDs to contiguous indices (one per node table)
- `{table_name}_metadata`: Global graph metadata (node count, edge count, directed flag)
- `schema.cypher`: A cypher schema that a graph database can mount without ingesting

## More information about graph-std and how it compares to Apache GraphAR

[Blog Post](https://adsharma.github.io/graph-archiving/)
