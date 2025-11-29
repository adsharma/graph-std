#!/usr/bin/env python3
"""
Script to convert graph data from DuckDB to CSR (Compressed Sparse Row) format.

This script reads graph data from a DuckDB database containing an edges table
with source and target columns representing edges, and converts it to CSR format for
efficient processing with NetworkKit.

The conversion process:
1. Reads graph data from DuckDB (edges table with source, target columns)
2. Handles sparse node IDs by creating a dense mapping (original_id -> csr_index)
3. Converts edges to CSR (Compressed Sparse Row) format
4. Pre-sorts edges by source using DuckDB for memory efficiency
5. Saves CSR data and node mapping to DuckDB for reuse
6. Exports to parquet format and generates schema.cypher for ladybugdb

Key Features:
- Memory efficient: Uses database-level sorting and PyArrow for large graph processing
- Handles sparse node IDs: Works with any node ID range (e.g., 1000, 5000, 9999)
- Scalable: Optimized for large graphs using DuckDB's efficient sorting
- Multi-table support: Processes multiple node/edge tables (prefix: nodes*, edges*)

Usage Examples:
    # Convert edges in karate_random.duckdb to CSR format and save to csr_graph.db
    python convert_csr.py --source-db karate_random.duckdb --output-db csr_graph.db

    # Convert with limited data for testing
    python convert_csr.py --source-db karate_random.duckdb --test --limit 50000 --output-db test.db
"""

import argparse
from pathlib import Path

import duckdb


def get_node_and_edge_tables(
    con, db_alias: str = "orig"
) -> tuple[list[str], list[str]]:
    """
    Discover node and edge tables in the source database.

    Tables starting with 'nodes' are considered node tables.
    Tables starting with 'edges' are considered edge tables.

    Returns:
        Tuple of (node_table_names, edge_table_names)
    """
    result = con.execute(
        f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_alias}'"
    ).fetchall()
    all_tables = [row[0] for row in result]

    node_tables = [t for t in all_tables if t.startswith("nodes")]
    edge_tables = [t for t in all_tables if t.startswith("edges")]

    return node_tables, edge_tables


def duckdb_type_to_cypher_type(duckdb_type: str) -> str:
    """Convert DuckDB column type to Cypher/Kuzu type."""
    duckdb_type = duckdb_type.upper()
    type_map = {
        "BIGINT": "INT64",
        "INTEGER": "INT32",
        "SMALLINT": "INT16",
        "TINYINT": "INT8",
        "HUGEINT": "INT128",
        "UBIGINT": "UINT64",
        "UINTEGER": "UINT32",
        "USMALLINT": "UINT16",
        "UTINYINT": "UINT8",
        "DOUBLE": "DOUBLE",
        "FLOAT": "FLOAT",
        "REAL": "FLOAT",
        "BOOLEAN": "BOOL",
        "VARCHAR": "STRING",
        "TEXT": "STRING",
        "CHAR": "STRING",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "TIME": "TIME",
        "BLOB": "BLOB",
    }
    # Handle parameterized types like DECIMAL(10,2)
    base_type = duckdb_type.split("(")[0].strip()
    return type_map.get(base_type, "STRING")


def generate_schema_cypher(
    con,
    csr_table_name: str,
    node_tables: list[str],
    edge_tables: list[str],
    parquet_dir: Path,
) -> str:
    """
    Generate schema.cypher content for ladybugdb.

    Args:
        con: DuckDB connection
        csr_table_name: Prefix for CSR tables
        node_tables: List of original node table names
        edge_tables: List of original edge table names
        parquet_dir: Path to the parquet output directory (for storage path)

    Returns:
        String containing the schema.cypher content
    """
    lines = []

    # Compute relative storage path (e.g., './karate_csr/karate_random')
    storage_path = f"./{parquet_dir.name}/{csr_table_name}"

    # Helper to derive display name from table name
    # nodes => nodes, nodes_person => person, nodes_foo => foo
    def get_node_display_name(table_name: str) -> str:
        if table_name == "nodes":
            return "nodes"
        elif table_name.startswith("nodes_"):
            return table_name[6:]  # Remove "nodes_" prefix
        return table_name

    def get_edge_display_name(table_name: str) -> str:
        if table_name == "edges":
            return "edges"
        elif table_name.startswith("edges_"):
            return table_name[6:]  # Remove "edges_" prefix
        return table_name

    # Build mapping of original table names to display names
    node_display_names = {nt: get_node_display_name(nt) for nt in node_tables}

    # Generate NODE TABLE definitions for each node table
    for node_table in node_tables:
        table_name = f"{csr_table_name}_{node_table}"
        try:
            cols = con.execute(f"DESCRIBE {table_name}").fetchall()
            col_defs = []
            pk_col = None
            for col in cols:
                col_name, col_type = col[0], col[1]
                cypher_type = duckdb_type_to_cypher_type(col_type)
                col_defs.append(f"{col_name} {cypher_type}")
                # First column is typically the primary key
                if pk_col is None:
                    pk_col = col_name

            cols_str = ", ".join(col_defs)
            display_name = node_display_names[node_table]
            lines.append(
                f"CREATE NODE TABLE {display_name}({cols_str}, PRIMARY KEY({pk_col})) WITH (storage = '{storage_path}');"
            )
        except Exception as e:
            print(
                f"Warning: Could not generate schema for node table {table_name}: {e}"
            )

    # Generate REL TABLE definitions for each edge table
    for edge_table in edge_tables:
        # Determine source and target node tables
        if node_tables:
            src_table = node_display_names[node_tables[0]]
            dst_table = src_table
        else:
            src_table = "nodes"
            dst_table = "nodes"

        rel_name = get_edge_display_name(edge_table)
        lines.append(
            f"CREATE REL TABLE {rel_name}(FROM {src_table} TO {dst_table}, weight DOUBLE) WITH (storage = '{storage_path}');"
        )

    return "\n".join(lines) + "\n"


def export_to_parquet_and_cypher(
    con,
    output_db_path: str,
    csr_table_name: str,
    node_tables: list[str],
    edge_tables: list[str],
) -> None:
    """
    Export all tables to parquet format and generate schema.cypher.

    Args:
        con: DuckDB connection
        output_db_path: Path to output DuckDB database
        csr_table_name: Prefix for CSR tables
        node_tables: List of original node table names
        edge_tables: List of original edge table names
    """
    print("\n=== Exporting to Parquet and Generating schema.cypher ===")

    # Create output directory next to the database
    output_path = Path(output_db_path)
    parquet_dir = output_path.parent / output_path.stem
    parquet_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parquet output directory: {parquet_dir}")

    # Get all tables to export
    result = con.execute("SHOW TABLES").fetchall()
    all_tables = [row[0] for row in result]

    # Export each table to parquet
    for table_name in all_tables:
        parquet_file = parquet_dir / f"{table_name}.parquet"
        con.execute(f"COPY {table_name} TO '{parquet_file}' (FORMAT 'parquet')")
        print(f"  Exported: {table_name} -> {parquet_file.name}")

    # Generate schema.cypher
    schema_cypher = generate_schema_cypher(
        con, csr_table_name, node_tables, edge_tables, parquet_dir
    )
    schema_file = parquet_dir / "schema.cypher"
    schema_file.write_text(schema_cypher)
    print(f"  Generated: {schema_file.name}")

    # Remove old SQL files if they exist
    for old_file in ["schema.sql", "load.sql"]:
        old_path = parquet_dir / old_file
        if old_path.exists():
            old_path.unlink()
            print(f"  Removed: {old_file}")

    print(f"✓ Export complete. Files saved to: {parquet_dir}")


def create_csr_graph_to_duckdb(
    source_db_path: str,
    output_db_path: str,
    limit_rels: int | None = None,
    directed: bool = False,
    csr_table_name: str = "csr_graph",
    node_table: str | None = None,
    edge_table: str | None = None,
) -> None:
    """
    Create CSR graph data and save to DuckDB using optimized SQL approach.

    Args:
        source_db_path: Path to source DuckDB with edges table
        output_db_path: Path to output DuckDB for CSR data
        limit_rels: Limit number of relationships for testing
        directed: Whether graph is directed
        csr_table_name: Name of table to store CSR data
        node_table: Specific node table to use (default: auto-discover)
        edge_table: Specific edge table to use (default: auto-discover)
    """
    print("\n=== Creating CSR Graph Data (Optimized SQL Approach) ===")

    # Connect to a fresh DuckDB database for output
    con = duckdb.connect(output_db_path)

    try:
        print("Step 0: Loading edges and nodes from original DB into new DB...")

        # Import the edges table from the original database
        con.execute(f"ATTACH '{source_db_path}' AS orig;")

        # Discover node and edge tables
        node_tables, edge_tables = get_node_and_edge_tables(con, "orig")

        # Use specified tables or discovered ones
        if node_table:
            node_tables = [node_table] if node_table in node_tables else []
        if edge_table:
            edge_tables = [edge_table] if edge_table in edge_tables else []

        if not edge_tables:
            raise ValueError(
                "No edge tables found in source database (tables must start with 'edges')"
            )

        print(f"Discovered node tables: {node_tables}")
        print(f"Discovered edge tables: {edge_tables}")

        # Copy all node tables with proper prefixing
        for nt in node_tables:
            try:
                con.execute(
                    f"CREATE TABLE {csr_table_name}_{nt} AS SELECT * FROM orig.{nt};"
                )
                print(f"  Copied node table: {nt} -> {csr_table_name}_{nt}")
            except Exception as e:
                print(f"Warning: Could not copy node table {nt}: {e}")

        # Build combined relations from all edge tables
        relations_queries = []
        for et in edge_tables:
            if limit_rels:
                limit_per_table = limit_rels // len(edge_tables)
                if directed:
                    relations_queries.append(
                        f"SELECT source, target FROM orig.{et} WHERE source != target LIMIT {limit_per_table}"
                    )
                else:
                    relations_queries.append(
                        f"""
                        WITH limited AS (
                            SELECT source, target FROM orig.{et} WHERE source != target LIMIT {limit_per_table}
                        )
                        SELECT source, target FROM limited
                        UNION ALL
                        SELECT target AS source, source AS target FROM limited
                        """
                    )
            else:
                if directed:
                    relations_queries.append(
                        f"SELECT source, target FROM orig.{et} WHERE source != target"
                    )
                else:
                    relations_queries.append(
                        f"""
                        SELECT source, target FROM orig.{et} WHERE source != target
                        UNION ALL
                        SELECT target AS source, source AS target FROM orig.{et} WHERE source != target
                        """
                    )

        # Combine all edge tables
        combined_query = " UNION ALL ".join(f"({q})" for q in relations_queries)
        con.execute(f"CREATE TABLE relations AS {combined_query};")

        if limit_rels:
            print(
                f"Using limited dataset: ~{limit_rels} edges total across {len(edge_tables)} edge table(s)"
            )

        print("Step 1: Creating id_mapping for contiguous node IDs...")

        # Create mapping from original IDs to 0-based contiguous IDs - rename to match existing schema
        # The order of columns is significant: csr_index first, original_node_id second
        con.execute(
            f"""
            CREATE TABLE {csr_table_name}_node_mapping AS
            SELECT
                row_number() OVER (ORDER BY node) - 1 AS csr_index,
                node AS original_node_id
            FROM (
                SELECT DISTINCT source AS node FROM relations
                UNION
                SELECT DISTINCT target AS node FROM relations
            )
            ORDER BY csr_index;
        """
        )

        # Index for fast lookup by original_id
        con.execute(
            f"CREATE UNIQUE INDEX idx_orig_id ON {csr_table_name}_node_mapping(original_node_id);"
        )

        result = con.execute(
            f"SELECT COUNT(*) FROM {csr_table_name}_node_mapping"
        ).fetchone()
        num_nodes = result[0] if result else 0
        print(f"Number of unique nodes: {num_nodes:,}")

        print("Step 2: Mapping edges to contiguous IDs...")

        con.execute(
            """
            CREATE TABLE relations_mapped AS
            SELECT
                m1.csr_index AS src,
                m2.csr_index AS dst
            FROM relations
            JOIN {}_node_mapping m1 ON relations.source = m1.original_node_id
            JOIN {}_node_mapping m2 ON relations.target = m2.original_node_id
        """.format(
                csr_table_name, csr_table_name
            )
        )

        result = con.execute("SELECT COUNT(*) FROM relations_mapped").fetchone()
        total_edges = result[0] if result else 0
        print(f"Total edges: {total_edges:,}")

        print("Step 3: Building csr_indptr (size: num_nodes + 1)...")

        # Build row_ptr: cumulative sum of degrees, including node IDs with zero degree
        con.execute(
            f"""
            CREATE TABLE {csr_table_name}_indptr AS
            WITH node_range AS (
                SELECT unnest(range(0, {num_nodes})) AS node_id
            ),
            degrees AS (
                SELECT src, COUNT(*) AS deg
                FROM relations_mapped
                GROUP BY src
            ),
            cumulative AS (
                SELECT
                    node_range.node_id,
                    COALESCE(SUM(degrees.deg) OVER (ORDER BY node_range.node_id ROWS UNBOUNDED PRECEDING), 0) AS ptr
                FROM node_range
                LEFT JOIN degrees ON node_range.node_id = degrees.src
            )
            SELECT ptr FROM cumulative
            ORDER BY node_id;

            -- Now prepend a 0 and append the last ptr again? No — we want:
            -- row_ptr[0] = 0
            -- row_ptr[i] = cumulative degree up to node i-1
            -- So we insert 0 at the beginning
        """
        )

        # Recreate csr_indptr with leading zero
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {csr_table_name}_indptr AS
            SELECT 0::BIGINT AS ptr  -- First element is 0
            UNION ALL
            SELECT ptr::int64 FROM {csr_table_name}_indptr
            ORDER BY ptr;
        """
        )

        # Validate size
        result = con.execute(f"SELECT COUNT(*) FROM {csr_table_name}_indptr").fetchone()
        indptr_size = result[0] if result else 0
        assert (
            indptr_size == num_nodes + 1
        ), f"csr_indptr should have {num_nodes + 1} entries, got {indptr_size}"

        print(f"csr_indptr created with {indptr_size} entries.")

        print("Step 4: Building csr_indices (sorted by src, then dst)...")

        # Create the column index array
        con.execute(
            f"""
            CREATE TABLE {csr_table_name}_indices AS
            SELECT dst AS target, 1.0::double as weight
            FROM relations_mapped
            ORDER BY src, dst;
        """
        )

        result = con.execute(
            f"SELECT COUNT(*) FROM {csr_table_name}_indices"
        ).fetchone()
        indices_size = result[0] if result else 0
        assert indices_size == total_edges, "csr_indices count mismatch"

        print(f"csr_indices created with {indices_size} entries.")

        print("Step 5: Creating metadata table...")

        # Create metadata table to match existing schema
        con.execute(
            f"""
            CREATE TABLE {csr_table_name}_metadata AS
            SELECT
                {num_nodes} AS n_nodes,
                {total_edges} AS n_edges,
                {directed} AS directed
        """
        )

        print("Step 6: Dropping temporary tables and indices...")

        # Drop intermediate tables
        con.execute("DROP TABLE IF EXISTS relations;")
        con.execute("DROP TABLE IF EXISTS relations_mapped;")

        # Drop indices
        con.execute("DROP INDEX IF EXISTS idx_orig_id;")
        con.execute("DROP INDEX IF EXISTS idx_mapped_id;")

        print("✅ CSR format built and cleaned up. Final tables:")
        print(f"  - {csr_table_name}_node_mapping (orig_id → mapped_id)")
        print(f"  - {csr_table_name}_indptr (array of size N+1)")
        print(f"  - {csr_table_name}_indices (array of size E)")
        print(f"  - {csr_table_name}_metadata (graph metadata)")

        # Optional: Quick validation
        result0 = con.execute(
            f"SELECT ptr FROM {csr_table_name}_indptr ORDER BY ptr LIMIT 1"
        ).fetchone()
        r0 = result0[0] if result0 else 0
        resultN = con.execute(
            f"SELECT ptr FROM {csr_table_name}_indptr ORDER BY ptr DESC LIMIT 1"
        ).fetchone()
        rN = resultN[0] if resultN else 0
        print(f"csr_indptr[0] = {r0}, csr_indptr[N] = {rN}")
        if rN == total_edges:
            print("✔️  CSR structure validated: last ptr == total edges")

        print(f"✓ Built CSR format: {num_nodes} nodes, {total_edges} edges")
        print(f"✓ Saved CSR graph data to {output_db_path}")

        # Export to parquet and generate schema.cypher
        export_to_parquet_and_cypher(
            con, output_db_path, csr_table_name, node_tables, edge_tables
        )

    except Exception as e:
        print(f"Error building CSR format: {e}")
        raise
    finally:
        con.close()

    print(f"\nAll data saved to: {output_db_path}")


def main():
    """Main function to convert DuckDB edges to CSR format."""
    parser = argparse.ArgumentParser(
        description="Convert graph data from DuckDB to CSR format"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="karate_random.duckdb",
        help="Source DuckDB database path (default: karate_random.duckdb)",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="csr_graph.db",
        help="Output DuckDB database path (default: csr_graph.db)",
    )
    parser.add_argument(
        "--csr-table",
        type=str,
        default="csr_graph",
        help="Table name prefix for CSR data (default: csr_graph)",
    )
    parser.add_argument(
        "--node-table",
        type=str,
        default=None,
        help="Specific node table to use (default: auto-discover tables starting with 'nodes')",
    )
    parser.add_argument(
        "--edge-table",
        type=str,
        default=None,
        help="Specific edge table to use (default: auto-discover tables starting with 'edges')",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Number of edges to use in test mode (default: 50000)",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat graph as directed (default: undirected)",
    )

    args = parser.parse_args()

    print("=== DuckDB to CSR Format Converter ===\n")

    # Configuration
    source_db_path = args.source_db  # DuckDB source

    # Create CSR graph
    test_limit = args.limit if args.test else None

    if test_limit:
        print(f"Creating CSR graph in TEST MODE with limit: {test_limit} edges")
    else:
        print("Creating CSR graph on FULL DATASET")

    print(f"Source database: {source_db_path}")
    print(f"CSR output database: {args.output_db}")
    print(f"CSR table prefix: {args.csr_table}")
    print(f"Directed: {args.directed}")
    if args.node_table:
        print(f"Node table filter: {args.node_table}")
    if args.edge_table:
        print(f"Edge table filter: {args.edge_table}")

    create_csr_graph_to_duckdb(
        source_db_path=source_db_path,
        output_db_path=args.output_db,
        limit_rels=test_limit,
        directed=args.directed,
        csr_table_name=args.csr_table,
        node_table=args.node_table,
        edge_table=args.edge_table,
    )

    print("\n=== Conversion Completed Successfully! ===")
    print(f"CSR graph data saved to: {args.output_db}")


if __name__ == "__main__":
    main()
