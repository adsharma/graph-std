CREATE NODE TABLE nodes(id INT64, club STRING, PRIMARY KEY(id)) WITH (storage = './karate_csr/karate');
CREATE REL TABLE edges(FROM nodes TO nodes) WITH (storage = './karate_csr/karate');
