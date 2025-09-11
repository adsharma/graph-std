COPY karate_random_indices FROM 'karate_csr/karate_random_indices.parquet' (FORMAT 'parquet');
COPY karate_random_indptr FROM 'karate_csr/karate_random_indptr.parquet' (FORMAT 'parquet');
COPY karate_random_metadata FROM 'karate_csr/karate_random_metadata.parquet' (FORMAT 'parquet');
COPY karate_random_nodes FROM 'karate_csr/karate_random_nodes.parquet' (FORMAT 'parquet');
COPY karate_random_node_mapping FROM 'karate_csr/karate_random_node_mapping.parquet' (FORMAT 'parquet');
