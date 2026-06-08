.PHONY: mamba pip clean

mamba:
	mamba env create -f environment.yaml

update:
	mamba env update --file environment.yaml --prune

pip:
	pip install ipykernel

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	mamba env remove -n coil-sketching-4d
