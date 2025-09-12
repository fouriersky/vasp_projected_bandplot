# vasp-projected-band-plotter

This project is designed to visualize band structures from VASP calculations, using projection weights to indicate contributions from different atomic orbitals. The visualization is achieved through a color gradient on the band structure plot, where the intensity of the color represents the projection weight.

## Features

- **Input Files**: The project takes three types of input files:
  - **PROCAR**: Contains the projection data.
  - **POSCAR**: Contains the atomic structure information.
  - **OUTCAR**: Contains the energy and k-point data.

- **Visualization**: The band structure is plotted with colors representing the projection weights, allowing for an intuitive understanding of the contributions from different atoms or orbitals.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/fouriersky/vasp-bandplot.git
conda activate env_name
cd vasp-projected-band-plotter/
python -m pip install .
```
## Testing

The project includes unit tests to ensure the functionality of the input/output operations, processing, and plotting. To run the tests, run the `run.py` in current directory, it should generate `test.png` in current directory.

```bash
cd ./test/
python -u run.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
