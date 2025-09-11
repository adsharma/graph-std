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

Key Features:
- Memory efficient: Uses database-level sorting and PyArrow for large graph processing
- Handles sparse node IDs: Works with any node ID range (e.g., 1000, 5000, 9999)
- Scalable: Optimized for large graphs using DuckDB's efficient sorting

Usage Examples:
    # Convert edges in karate_random.duckdb to CSR format and save to csr_graph.db
    python convert_csr.py --source-db karate_random.duckdb --output-db csr_graph.db

    # Convert with limited data for testing
    python convert_csr.py --source-db karate_random.duckdb --test --limit 50000 --output-db test.db
"""

import duckdb
import argparse
import pyarrow as pa
import time


def create_csr_graph_to_duckdb(
    source_db_path: str,
    output_db_path: str,
    limit_rels: int | None = None,
    directed: bool = False,
    csr_table_name: str = "csr_graph",
) -> None:
    """
    Create CSR graph data and save to DuckDB using optimized SQL approach.

    Args:
        source_db_path: Path to source DuckDB with edges table
        output_db_path: Path to output DuckDB for CSR data
        limit_rels: Limit number of relationships for testing
        directed: Whether graph is directed
        csr_table_name: Name of table to store CSR data
    """
    print(f"\n=== Creating CSR Graph Data (Optimized SQL Approach) ===")

    # Connect to a fresh DuckDB database for output
    con = duckdb.connect(output_db_path)

    try:
        print("Step 0: Loading edges from original DB into new DB...")

        # Import the edges table from the original database
        con.execute(f"ATTACH '{source_db_path}' AS orig;")

        # Handle limited edges for testing
        if limit_rels:
            print(f"Using limited dataset: {limit_rels} edges")
            if directed:
                con.execute(
                    f"CREATE TABLE relations AS SELECT source, target FROM orig.edges WHERE source != target LIMIT {limit_rels};"
                )
            else:
                # For undirected, we need to double the edges
                con.execute(
                    f"""
                    CREATE TABLE relations AS 
                    WITH limited AS (
                        SELECT source, target FROM orig.edges WHERE source != target LIMIT {limit_rels}
                    )
                    SELECT source, target FROM limited
                    UNION ALL
                    SELECT target AS source, source AS target FROM limited;
                """
                )
        else:
            if directed:
                con.execute(
                    "CREATE TABLE relations AS SELECT source, target FROM orig.edges WHERE source != target;"
                )
            else:
                # For undirected, create bidirectional edges
                con.execute(
                    """
                    CREATE TABLE relations AS 
                    SELECT source, target FROM orig.edges WHERE source != target
                    UNION ALL
                    SELECT target AS source, source AS target FROM orig.edges WHERE source != target;
                """
                )

        print("Step 1: Creating id_mapping for contiguous node IDs...")

        # Create mapping from original IDs to 0-based contiguous IDs - rename to match existing schema
        con.execute(
            f"""
            CREATE TABLE {csr_table_name}_node_mapping AS
            SELECT 
                node AS original_node_id,
                row_number() OVER (ORDER BY node) - 1 AS csr_index
            FROM (
                SELECT DISTINCT source AS node FROM relations
                UNION
                SELECT DISTINCT target AS node FROM relations
            )
        """
        )

        # Index for fast lookup
        con.execute(
            f"CREATE UNIQUE INDEX idx_orig_id ON {csr_table_name}_node_mapping(original_node_id);"
        )
        con.execute(
            f"CREATE UNIQUE INDEX idx_mapped_id ON {csr_table_name}_node_mapping(csr_index);"
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
            SELECT ptr FROM {csr_table_name}_indptr
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
            SELECT dst AS target
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

    create_csr_graph_to_duckdb(
        source_db_path=source_db_path,
        output_db_path=args.output_db,
        limit_rels=test_limit,
        directed=args.directed,
        csr_table_name=args.csr_table,
    )

    print(f"\n=== Conversion Completed Successfully! ===")
    print(f"CSR graph data saved to: {args.output_db}")


if __name__ == "__main__":
    main()
