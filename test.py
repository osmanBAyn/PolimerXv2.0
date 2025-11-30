        if True:
            repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
            tg_data = load_dataset(repo_id,split="Tg")
            df = tg_data.to_pandas()
            # Sütun adının 'p_smiles' veya 'smiles' olduğundan emin ol
            col_name = 'p_smiles' if 'p_smiles' in df.columns else 'smiles'
            initial_pop = df[col_name].tolist()