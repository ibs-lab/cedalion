EXAMPLE_NOTEBOOKS = getting_started_io/00_test_installation.ipynb \
					getting_started_io/10_xarray_datastructs_fnirs.ipynb \
					getting_started_io/11_recording_container.ipynb \
					getting_started_io/13_data_structures_intro.ipynb \
					getting_started_io/34_store_hrfs_in_snirf_file.ipynb \
					getting_started_io/14_snirf2bids.ipynb \
					plots_visualization/12_plots_example.ipynb \
					machine_learning/50_finger_tapping_lda_classification.ipynb \
					machine_learning/52_ica_erbm_fingertapping_example.ipynb \
					machine_learning/53_constrained_ICA_example.ipynb \
					modeling/31_glm_basis_functions.ipynb \
					modeling/32_glm_fingertapping_example.ipynb \
					modeling/33_glm_illustrative_example.ipynb \
					modeling/35_statsmodels_overview.ipynb \
					modeling/36_glm_workshop.ipynb \
					signal_quality/20_scalp_coupling_index.ipynb \
					signal_quality/21_data_quality_and_pruning.ipynb \
					signal_quality/22_motion_artefacts_and_correction.ipynb \
					signal_quality/24_downweighting_noisy_channels.ipynb \
					signal_quality/25_intro_quality_workshop.ipynb \
					head_models/40_image_reconstruction.ipynb \
					head_models/41_photogrammetric_optode_coregistration.ipynb \
					head_models/42_1010_system.ipynb \
					head_models/43_crs_and_headmodel.ipynb \
					head_models/44_head_models.ipynb \
					head_models/45_parcel_sensitivity.ipynb \
					head_models/46_precompute_fluence.ipynb \
					head_models/48_headmodel_landmarks_verification.ipynb \
					augmentation/61_synthetic_artifacts_example.ipynb \
					augmentation/62_synthetic_hrfs_example.ipynb \
					physio/71_ampd_heartbeat.ipynb \
					tutorial/1_heads_and_fwm.ipynb \
					tutorial/2_photogrammetry.ipynb \
					tutorial/3_signal_processing.ipynb \
					tutorial/4_model_driven_analysis.ipynb \
					tutorial/5_image_reconstruction.ipynb \
					tutorial/6_data_driven_analysis.ipynb \
					tutorial/7_data_augmentation.ipynb

.PHONY: notebooks clean

notebooks: $(EXAMPLE_NOTEBOOKS)

clean:
	rm -f $(EXAMPLE_NOTEBOOKS)

$(EXAMPLE_NOTEBOOKS): %.ipynb : ../../examples/%.ipynb
	jupyter nbconvert --ExecutePreprocessor.store_widget_state=True --to notebook \
	      --execute --output-dir=$(dir $@) $<
