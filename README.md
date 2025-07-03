# coil-sketching-4D

**4D MRI Reconstruction via Coil Sketching**  
**Author:** Joseph Plummer

This repository contains experimental code for coil sketching-based reconstruction of 4D MRI datasets. It builds upon the excellent open-source framework developed by Julio Oscanoa et al., available at: [https://github.com/julioscanoa/sketching_mri](https://github.com/julioscanoa/sketching_mri). We are grateful for their foundational work.

> ⚠️ **Note:** This is an active research repository and remains under development. Contributions and feedback are welcome.

---

## Installation

To set up the environment and run the experiments, execute the following commands in order:

```bash
mamba update -n base -c defaults mamba
make mamba
mamba activate coil-sketching-4d
make pip
```

### Troubleshooting

This repository was developed and tested on systems with NVIDIA GPUs. If you are installing on a CPU-only system or one without CUDA support, remove the following GPU-specific dependencies from `environment.yaml`:

- `cudnn`
- `nccl`
- `cupy`

---

## Development Workflow

If you intend to modify or extend the code, please work from a new development branch:

```bash
git checkout -b my-feature-branch
```

Pull requests and issue submissions are encouraged.

---

## Uninstallation

To clean up the environment:

```bash
mamba activate
make clean
```

---

## Citation and Publication

A manuscript is currently in preparation. A DOI will be provided upon publication.
