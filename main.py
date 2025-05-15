from pathlib import Path
from dataset_loader import Dataset
from mirp import extract_features
import cv2 as cv
import pandas as pd

from config.config import DATASETS, DATASET_PATHS


if __name__ =="__main__":

    d = Dataset(DATASET_PATHS[DATASETS.SCI_TEST])


    layers = [[],[],[]]
    for i, scene in enumerate(d.scenes):
        print(f"{i+1}/{len(d.scenes)}")

        gray_imgs = scene.dark
        masks = scene.mask

        for layer_idx in range(len(gray_imgs)):
            if gray_imgs[layer_idx] is None:
                continue

            features = extract_features(
                image=gray_imgs[layer_idx],
                mask=masks,
                intensity_normalisation="standardisation",
                base_discretisation_method="fixed_bin_number",
                base_discretisation_n_bins=32
            ) 

            layers[layer_idx].extend([f.assign(scene_id = scene.scene_id) for f in features])


    for layer_idx in range(len(gray_imgs)):
        if len(layers[layer_idx]) > 0:
            result = pd.concat(layers[layer_idx], ignore_index=True)
        else:
            result = pd.DataFrame([])
        result.to_csv(f'outputs/{DATASETS.SCI_TEST.name}_layer_{layer_idx+1}.csv', index=False)  
              


# ['sample_name', 'image_file_name', 'image_directory', 'image_study_date', 'image_study_description', 'image_series_description', 'image_series_instance_uid', 'image_modality', 'image_pet_suv_type', 'image_mask_label', 'image_mask_file_name', 'image_mask_directory', 'image_mask_series_description', 'image_mask_series_instance_uid', 'image_settings_id', 'image_voxel_size_x', 'image_voxel_size_y', 'image_voxel_size_z', 'image_noise_level', 'image_noise_iteration_id', 'image_rotation_angle', 'image_translation_x', 'image_translation_y', 'image_translation_z', 'image_mask_randomise_id', 'image_mask_adapt_size', 'morph_volume', 'morph_vol_approx', 'morph_area_mesh', 'morph_av', 'morph_comp_1', 'morph_comp_2', 'morph_sph_dispr', 'morph_sphericity', 'morph_asphericity', 'morph_com', 'morph_integ_int', 'morph_diam', 'morph_vol_dens_conv_hull', 'morph_area_dens_conv_hull', 'morph_vol_dens_aabb', 'morph_area_dens_aabb', 'morph_vol_dens_ombb', 'morph_area_dens_ombb', 'morph_vol_dens_mvee', 'morph_area_dens_mvee', 'morph_pca_maj_axis', 'morph_pca_min_axis', 'morph_pca_least_axis', 'morph_pca_elongation', 'morph_pca_flatness', 'morph_vol_dens_aee', 'morph_area_dens_aee', 'morph_moran_i', 'morph_geary_c', 'loc_peak_loc', 'loc_peak_glob', 'ivh_v10', 'ivh_v25', 'ivh_v50', 'ivh_v75', 'ivh_v90', 'ivh_i10', 'ivh_i25', 'ivh_i50', 'ivh_i75', 'ivh_i90', 'ivh_diff_v10_v90', 'ivh_diff_v25_v75', 'ivh_diff_i10_i90', 'ivh_diff_i25_i75', 'ivh_auc', 'stat_mean', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median', 'stat_min', 'stat_p10', 'stat_p90', 'stat_max', 'stat_iqr', 'stat_range', 'stat_mad', 'stat_rmad', 'stat_medad', 'stat_cov', 'stat_qcod', 'stat_energy', 'stat_rms', 'ih_mean_fbn_n32', 'ih_var_fbn_n32', 'ih_skew_fbn_n32', 'ih_kurt_fbn_n32', 'ih_median_fbn_n32', 'ih_min_fbn_n32', 'ih_p10_fbn_n32', 'ih_p90_fbn_n32', 'ih_max_fbn_n32', 'ih_mode_fbn_n32', 'ih_iqr_fbn_n32', 'ih_range_fbn_n32', 'ih_mad_fbn_n32', 'ih_rmad_fbn_n32', 'ih_medad_fbn_n32', 'ih_cov_fbn_n32', 'ih_qcod_fbn_n32', 'ih_entropy_fbn_n32', 'ih_uniformity_fbn_n32', 'ih_max_grad_fbn_n32', 'ih_max_grad_g_fbn_n32', 'ih_min_grad_fbn_n32', 'ih_min_grad_g_fbn_n32', 'cm_joint_max_d1_3d_v_mrg_fbn_n32', 'cm_joint_avg_d1_3d_v_mrg_fbn_n32', 'cm_joint_var_d1_3d_v_mrg_fbn_n32', 'cm_joint_entr_d1_3d_v_mrg_fbn_n32', 'cm_diff_avg_d1_3d_v_mrg_fbn_n32', 'cm_diff_var_d1_3d_v_mrg_fbn_n32', 'cm_diff_entr_d1_3d_v_mrg_fbn_n32', 'cm_sum_avg_d1_3d_v_mrg_fbn_n32', 'cm_sum_var_d1_3d_v_mrg_fbn_n32', 'cm_sum_entr_d1_3d_v_mrg_fbn_n32', 'cm_energy_d1_3d_v_mrg_fbn_n32', 'cm_contrast_d1_3d_v_mrg_fbn_n32', 'cm_dissimilarity_d1_3d_v_mrg_fbn_n32', 'cm_inv_diff_d1_3d_v_mrg_fbn_n32', 'cm_inv_diff_norm_d1_3d_v_mrg_fbn_n32', 'cm_inv_diff_mom_d1_3d_v_mrg_fbn_n32', 'cm_inv_diff_mom_norm_d1_3d_v_mrg_fbn_n32', 'cm_inv_var_d1_3d_v_mrg_fbn_n32', 'cm_corr_d1_3d_v_mrg_fbn_n32', 'cm_auto_corr_d1_3d_v_mrg_fbn_n32', 'cm_clust_tend_d1_3d_v_mrg_fbn_n32', 'cm_clust_shade_d1_3d_v_mrg_fbn_n32', 'cm_clust_prom_d1_3d_v_mrg_fbn_n32', 'cm_info_corr1_d1_3d_v_mrg_fbn_n32', 'cm_info_corr2_d1_3d_v_mrg_fbn_n32', 'rlm_sre_3d_v_mrg_fbn_n32', 'rlm_lre_3d_v_mrg_fbn_n32', 'rlm_lgre_3d_v_mrg_fbn_n32', 'rlm_hgre_3d_v_mrg_fbn_n32', 'rlm_srlge_3d_v_mrg_fbn_n32', 'rlm_srhge_3d_v_mrg_fbn_n32', 'rlm_lrlge_3d_v_mrg_fbn_n32', 'rlm_lrhge_3d_v_mrg_fbn_n32', 'rlm_glnu_3d_v_mrg_fbn_n32', 'rlm_glnu_norm_3d_v_mrg_fbn_n32', 'rlm_rlnu_3d_v_mrg_fbn_n32', 'rlm_rlnu_norm_3d_v_mrg_fbn_n32', 'rlm_r_perc_3d_v_mrg_fbn_n32', 'rlm_gl_var_3d_v_mrg_fbn_n32', 'rlm_rl_var_3d_v_mrg_fbn_n32', 'rlm_rl_entr_3d_v_mrg_fbn_n32', 'szm_sze_3d_fbn_n32', 'szm_lze_3d_fbn_n32', 'szm_lgze_3d_fbn_n32', 'szm_hgze_3d_fbn_n32', 'szm_szlge_3d_fbn_n32', 'szm_szhge_3d_fbn_n32', 'szm_lzlge_3d_fbn_n32', 'szm_lzhge_3d_fbn_n32', 'szm_glnu_3d_fbn_n32', 'szm_glnu_norm_3d_fbn_n32', 'szm_zsnu_3d_fbn_n32', 'szm_zsnu_norm_3d_fbn_n32', 'szm_z_perc_3d_fbn_n32', 'szm_gl_var_3d_fbn_n32', 'szm_zs_var_3d_fbn_n32', 'szm_zs_entr_3d_fbn_n32', 'dzm_sde_3d_fbn_n32', 'dzm_lde_3d_fbn_n32', 'dzm_lgze_3d_fbn_n32', 'dzm_hgze_3d_fbn_n32', 'dzm_sdlge_3d_fbn_n32', 'dzm_sdhge_3d_fbn_n32', 'dzm_ldlge_3d_fbn_n32', 'dzm_ldhge_3d_fbn_n32', 'dzm_glnu_3d_fbn_n32', 'dzm_glnu_norm_3d_fbn_n32', 'dzm_zdnu_3d_fbn_n32', 'dzm_zdnu_norm_3d_fbn_n32', 'dzm_z_perc_3d_fbn_n32', 'dzm_gl_var_3d_fbn_n32', 'dzm_zd_var_3d_fbn_n32', 'dzm_zd_entr_3d_fbn_n32', 'ngt_coarseness_3d_fbn_n32', 'ngt_contrast_3d_fbn_n32', 'ngt_busyness_3d_fbn_n32', 'ngt_complexity_3d_fbn_n32', 'ngt_strength_3d_fbn_n32', 'ngl_lde_d1_a0.0_3d_fbn_n32', 'ngl_hde_d1_a0.0_3d_fbn_n32', 'ngl_lgce_d1_a0.0_3d_fbn_n32', 'ngl_hgce_d1_a0.0_3d_fbn_n32', 'ngl_ldlge_d1_a0.0_3d_fbn_n32', 'ngl_ldhge_d1_a0.0_3d_fbn_n32', 'ngl_hdlge_d1_a0.0_3d_fbn_n32', 'ngl_hdhge_d1_a0.0_3d_fbn_n32', 'ngl_glnu_d1_a0.0_3d_fbn_n32', 'ngl_glnu_norm_d1_a0.0_3d_fbn_n32', 'ngl_dcnu_d1_a0.0_3d_fbn_n32', 'ngl_dcnu_norm_d1_a0.0_3d_fbn_n32', 'ngl_dc_perc_d1_a0.0_3d_fbn_n32', 'ngl_gl_var_d1_a0.0_3d_fbn_n32', 'ngl_dc_var_d1_a0.0_3d_fbn_n32', 'ngl_dc_entr_d1_a0.0_3d_fbn_n32', 'ngl_dc_energy_d1_a0.0_3d_fbn_n32']