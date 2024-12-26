import os

def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u ./src/benchmark/GNN_PPI/gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs))


if __name__ == "__main__":
    # description = "GOProteinGCN_148k_dfs"
    # ppi_path = "datasets/downstream_datasets/data/protein.actions.SHS148k.STRING.txt"
    # pseq_path = "datasets/downstream_datasets/data/protein.SHS148k.sequences.dictionary.tsv"
    # vec_path = "datasets/downstream_datasets/data/PPI_embeddings/protein_embedding_GOProteinGCN_shs148k.npy"

    # description = "GOProteinGCN_27k_dfs"
    # ppi_path = "datasets/downstream_datasets/data/protein.actions.SHS27k.STRING.txt"
    # pseq_path = "datasets/downstream_datasets/data/protein.SHS27k.sequences.dictionary.tsv"
    # vec_path = "datasets/downstream_datasets/data/PPI_embeddings/protein_embedding_GOProteinGCN_shs27k.npy"

    description = "GOProteinGCN_STRINGS_bfs"
    ppi_path = "datasets/downstream_datasets/data/9606.protein.actions.all_connected.txt"
    pseq_path = "datasets/downstream_datasets/data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "datasets/downstream_datasets/data/PPI_embeddings/protein_embedding_GOProteinGCN_STRING.npy"

    split_new = "True"      # This parameter enable to create a new train-test split
    split_mode = "bfs"

    train_valid_index_path = "datasets/downstream_datasets/data/ppi_splits/GOProteinGNN-STRINGS.bfs.json"

    use_lr_scheduler = "True"
    save_path = "gcn-output/ppi"
    graph_only_train = "False"

    batch_size = 1024
    epochs = 200

    run_func(description, ppi_path, pseq_path, vec_path, 
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs)
