from typing import Dict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from src.data import RiboNNDataModule
from src.model import RiboNN
from src.utils.helpers import extract_config


def predict_using_models_trained_in_one_fold(
    run_df: pd.DataFrame,
    config: Dict,
    dm: pl.LightningDataModule,
    cached_batches: list,
    device: torch.device,
    top_k_models_to_use: int = 5,
) -> pd.DataFrame:
    """Make predictions averaged across top_k_models listed in mlflow run_df.
    """
    # TE names
    if config['species'] == 'human':
        training_data_columns = "TE_108T,TE_12T,TE_A2780,TE_A549,TE_BJ,TE_BRx.142,TE_C643,TE_CRL.1634,TE_Calu.3,TE_Cybrid_Cells,TE_H1.hESC,TE_H1933,TE_H9.hESC,TE_HAP.1,TE_HCC_tumor,TE_HCC_adjancent_normal,TE_HCT116,TE_HEK293,TE_HEK293T,TE_HMECs,TE_HSB2,TE_HSPCs,TE_HeLa,TE_HeLa_S3,TE_HepG2,TE_Huh.7.5,TE_Huh7,TE_K562,TE_Kidney_normal_tissue,TE_LCL,TE_LuCaP.PDX,TE_MCF10A,TE_MCF10A.ER.Src,TE_MCF7,TE_MD55A3,TE_MDA.MB.231,TE_MM1.S,TE_MOLM.13,TE_Molt.3,TE_Mutu,TE_OSCC,TE_PANC1,TE_PATU.8902,TE_PC3,TE_PC9,TE_Primary_CD4._T.cells,TE_Primary_human_bronchial_epithelial_cells,TE_RD.CCL.136,TE_RPE.1,TE_SH.SY5Y,TE_SUM159PT,TE_SW480TetOnAPC,TE_T47D,TE_THP.1,TE_U.251,TE_U.343,TE_U2392,TE_U2OS,TE_Vero_6,TE_WI38,TE_WM902B,TE_WTC.11,TE_ZR75.1,TE_cardiac_fibroblasts,TE_ccRCC,TE_early_neurons,TE_fibroblast,TE_hESC,TE_human_brain_tumor,TE_iPSC.differentiated_dopamine_neurons,TE_megakaryocytes,TE_muscle_tissue,TE_neuronal_precursor_cells,TE_neurons,TE_normal_brain_tissue,TE_normal_prostate,TE_primary_macrophages,TE_skeletal_muscle"
    elif config['species'] == 'mouse':
        training_data_columns = "TE_3T3,TE_4T1,TE_A3-1_mESC,TE_B16_melanoma_cell,TE_Bone_marrow_derived_macrophage,TE_Bone_marrow_derived_primary_dendritic_cells,TE_Bone_marrow_derived_regulatory_dendritic_cells,TE_CD4_T-cells,TE_CGR8_mESC,TE_Cerebellum,TE_DRG_neuronal_culture,TE_Dorsal_section_of_lumbar_spinal_cord,TE_E14_mESC,TE_E14Tg2a_mESC,TE_ES_cell_derived_neurons,TE_Epiblast_like_cells,TE_Epidermal_basal_cells,TE_Fetal_cortex,TE_Flt3L-DC,TE_Forebrain,TE_HFSC,TE_Ileum,TE_J1_mESC,TE_KH2_mESC,TE_KRPC-A_cells_(from_gen._eng._murine_tumors),TE_MN1_cells,TE_Mammary_tissue,TE_Mouse_back_skins,TE_NSC,TE_Neuro2a,TE_Neurons_(DIV_8)_derived_from_CGR8_ES_cells,TE_P19,TE_Primary_Keratinocytes,TE_Primary_cortical_neurons,TE_Quadriceps_muscle,TE_R1_mESC,TE_R1/E_mESC,TE_RAW264.7,TE_RF8_mESC,TE_Spontaneously_Immortalized_Mouse_Keratinocyte_Cult,TE_Striatal_cells,TE_T-ALL,TE_brain,TE_dentate_gyrus,TE_duodenum,TE_embryonic_fibroblast,TE_embryonic_stem_cells,TE_epididymal_white_fat,TE_forelimbs,TE_gastrocnemius_tissue,TE_heart,TE_hippocampal,TE_interscapular_brown_fat,TE_kidney,TE_liver,TE_lung,TE_lymphoid_ba/f3_cells,TE_mouse_eye,TE_neural_tube,TE_neutrophils,TE_pancreas,TE_resting_state_T_cells,TE_skeletal_muscle,TE_skin_squamous_tumours_(skin_papilloma),TE_subcutaneous_white_fat,TE_testis,TE_tibialis_anterior_tissue,TE_v6.5_mESC"
    else:
        raise ValueError(f"Unknown species: {config['species']}")

    predicted_columns = training_data_columns.replace("TE_", "predicted_TE_").split(",")

    # Filter run_df to keep the top k models ranked by validation R2
    run_df = run_df.sort_values("metrics.val_r2", ascending=False).head(
        top_k_models_to_use
    )

    # Iterate over the models to make predictions
    predictions = []
    for run_id in run_df.run_id:
        # Create a new model
        model = RiboNN(**config)

        # Load the state dict
        local_state_dict_path = f"models/{config['species']}/{run_id}/state_dict.pth"
        model.load_state_dict(torch.load(local_state_dict_path))
        model.to(device)
        model.eval()

        with torch.no_grad():
            # Reuse the once-encoded batches (already on `device`) instead of rebuilding
            # and re-encoding the dataloader for every model — that encoding is identical
            # across models and is the dominant cost of the whole prediction.
            batched_predictions = [model(batch) for batch in cached_batches]

        if isinstance(batched_predictions, list):
            batched_predictions = torch.cat(batched_predictions, dim=0)

        predictions.append(batched_predictions.cpu().numpy())

    mean_prediction = np.stack(predictions, axis=-1).mean(axis=-1)

    df = pd.DataFrame(mean_prediction, columns=predicted_columns)

    df = pd.concat([dm.df, df], axis=1)

    return df


def predict_using_nested_cross_validation_models(
    input_path: str,
    species: str,
    run_df: str,
    top_k_models_to_use: int = 5,
    batch_size: int = 1024,
    num_workers: int = 4,
) -> pd.DataFrame:

    print(f"Making predictions using {species} models...")

    # Create data module
    config = extract_config(run_df, run_df.run_id[0])
    config["species"] = species
    config["max_utr5_len"] = 1_381  # used when training the model
    config["max_cds_utr3_len"] = 11_937  # used when training the model
    config["tx_info_path"] = input_path
    config["num_workers"] = num_workers
    config["test_batch_size"] = batch_size
    config["remove_extreme_txs"] = False
    config["target_column_pattern"] = None
    dm = RiboNNDataModule(config)

    # Encode the dataset ONCE and reuse across every fold/model. The per-nt one-hot
    # encoding in DataFrameDataset.__getitem__ is the dominant cost and is identical for
    # all models, so rebuilding the dataloader per model just re-pays it. Cached batches
    # are moved to `device` here so each model pass is a pure forward on resident tensors.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cached_batches = [b.to(device) for b in dm.predict_dataloader()]

    all_prediction_dfs = []
    for test_fold in np.sort(run_df["params.test_fold"].unique()):
        test_fold_str = str(test_fold)
        sub_run_df = run_df.query(
            "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
        ).reset_index(drop=True)
        prediction_df = predict_using_models_trained_in_one_fold(
            sub_run_df, config, dm, cached_batches, device, top_k_models_to_use
        )
        prediction_df["fold"] = int(test_fold)

        all_prediction_dfs.append(prediction_df)

    all_predictions = pd.concat(all_prediction_dfs, axis=0, ignore_index=True)

    return all_predictions


