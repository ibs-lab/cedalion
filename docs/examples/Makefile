EXAMPLE_NOTEBOOKS = getting_started_io/00_test_installation.ipynb \
                    getting_started_io/10_xarray_datastructs_fnirs.ipynb \
                    getting_started_io/11_recording_container.ipynb \
					getting_started_io/13_data_structures_intro.ipynb \
                    getting_started_io/34_store_hrfs_in_snirf_file.ipynb \
                    plots_visualization/12_plots_example.ipynb \
                    machine_learning/50_finger_tapping_lda_classification.ipynb \
                    modeling/31_glm_basis_functions.ipynb \
                    modeling/32_glm_fingertapping_example.ipynb \
                    modeling/33_glm_illustrative_example.ipynb \
                    signal_quality/20_scalp_coupling_index.ipynb \
                    signal_quality/21_data_quality_and_pruning.ipynb \
                    signal_quality/22_motion_artefacts_and_correction.ipynb \
                    head_models/40_image_reconstruction.ipynb \
                    head_models/41_photogrammetric_optode_coregistration.ipynb \
                    head_models/42_1010_system.ipynb \
                    head_models/43_crs_and_headmodel.ipynb

.PHONY: notebooks

notebooks: $(EXAMPLE_NOTEBOOKS)

$(EXAMPLE_NOTEBOOKS): %.ipynb : ../../examples/%.ipynb
	jupyter nbconvert --ExecutePreprocessor.store_widget_state=True --to notebook \
	      --execute --output-dir=$(dir $@) $<