EXAMPLE_NOTEBOOKS = xarray_datastructs_fnirs.ipynb \
					finger_tapping_lda_classification.ipynb \
					pruning_and_motion_artifacts.ipynb \
					photogrammetric_optode_coregistration.ipynb \
					scalp_coupling_index.ipynb \
					1010_system.ipynb \
					image_reconstruction.ipynb \
					store_hrfs_in_snirf_file.ipynb\
					recording.ipynb



.PHONY: notebooks

notebooks: $(EXAMPLE_NOTEBOOKS)

$(EXAMPLE_NOTEBOOKS) : %.ipynb : ../../examples/%.ipynb
	jupyter nbconvert --ExecutePreprocessor.store_widget_state=True --to notebook \
	      --execute --output-dir=. $^

clean:
	rm -f $(EXAMPLE_NOTEBOOKS)
