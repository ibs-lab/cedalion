EXAMPLE_NOTEBOOKS = 00_test_installation.ipynb \
					10_xarray_datastructs_fnirs.ipynb \
					11_recording_container.ipynb \
					12_plots_example.ipynb \
					13_data_structures_intro.ipynb \
					20_scalp_coupling_index.ipynb \
					21_data_quality_and_pruning.ipynb \
					22_motion_artefacts_and_correction.ipynb \
					31_glm_basis_functions.ipynb \
					32_glm_fingertapping_example.ipynb \
					33_glm_illustrative_example.ipynb \
					34_store_hrfs_in_snirf_file.ipynb \
					40_image_reconstruction.ipynb \
					41_photogrammetric_optode_coregistration.ipynb \
					42_1010_system.ipynb \
					43_crs_and_headmodel.ipynb \
					50_finger_tapping_lda_classification.ipynb
					
					


.PHONY: notebooks

notebooks: $(EXAMPLE_NOTEBOOKS)

$(EXAMPLE_NOTEBOOKS) : %.ipynb : ../../examples/%.ipynb
	jupyter nbconvert --ExecutePreprocessor.store_widget_state=True --to notebook \
	      --execute --output-dir=. $^

clean:
	rm -f $(EXAMPLE_NOTEBOOKS)
