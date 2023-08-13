import os


def run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all):
    os.system("python ./src/benchmark/GNN_PPI/gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all))


if __name__ == "__main__":
    description = "test-strings-bfs"

    ppi_path = "datasets/downstream_datasets/data/9606.protein.actions.all_connected.txt"
    pseq_path = "datasets/downstream_datasets/data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "datasets/downstream_datasets/data/PPI_embeddings/protein_embedding_GOProteinGCN_STRING.npy"

    index_path = "datasets/downstream_datasets/data/ppi_splits/GOProteinGNN-STRINGS.bfs.json"

    # path to checkpoint
    dir_name = "gcn-output/ppi/gnn_GOProteinGCN_STRINGS_bfs"
    gnn_model = dir_name + "/gnn_model_valid_best.ckpt"

    test_all = "True"

    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all)
