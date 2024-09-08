EXAMPLE_NOTEBOOKS = xarray_datastructs_fnirs.ipynb \
					recording.ipynb \
					finger_tapping_lda_classification.ipynb \
					scalp_coupling_index.ipynb \
					1010_system.ipynb \
					store_hrfs_in_snirf_file.ipynb \
					glm_basis_functions.ipynb \
					glm_illustrative_example.ipynb \
					glm_fingertapping_example.ipynb 
					plots_example.ipynb
					#image_reconstruction.ipynb \
					photogrammetric_optode_coregistration.ipynb \
					#pruning_and_motion_artifacts.ipynb \

.PHONY: notebooks

notebooks: $(EXAMPLE_NOTEBOOKS)

$(EXAMPLE_NOTEBOOKS) : %.ipynb : ../../examples/%.ipynb
	jupyter nbconvert --ExecutePreprocessor.store_widget_state=True --to notebook \
	      --execute --output-dir=. $^

clean:
	rm -f $(EXAMPLE_NOTEBOOKS)
