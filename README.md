# 🦏 rhino-magic

**rhino-magic** is a modular and extensible machine learning framework designed for rapid experimentation and deployment. It provides a structured approach to training and evaluating models, leveraging configuration files and utility scripts to streamline workflows.

---

## 📁 Project Structure

```
rhino-magic/
├── configs/              # Configuration files for experiments
├── evaluate_scripts/     # Scripts for model evaluation
├── jupyter_notebooks/    # Notebooks for data exploration and prototyping
├── rhino/                # Core library modules
├── rhino_magic.egg-info/ # Package metadata
├── train_scripts/        # Scripts for training models
├── trainer/              # Training logic and utilities
├── utils/                # Helper functions and utilities
├── work_dirs/            # Output directories for experiments (it is empty)
├── .gitignore            # Git ignore rules
├── memory_debug.log      # Memory usage logs
├── pyproject.toml        # Build system configuration
├── requirements.txt      # Python dependencies
└── setup.py              # Package installation script
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11
* Pytorch 2.2.1
* Diffusers 0.33.1
* Numpy >=1.24.4

* Recommended: Create a virtual environment

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yyhtbs-yye/rhino-magic.git
   cd rhino-magic
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install the package:

   ```bash
   pip install -e .
   ```

---

## ⚙️ Configuration

Experiment settings are managed through YAML files located in the `configs/` directory. These files define parameters such as model architecture, training hyperparameters, and dataset paths.

Example configuration file: `configs/example_config.yaml`

---

## ⚙️ Key Designs

This model mimics the torch lightning framework, but not compatible. It aims to support more complex but regulated training/inference logics. Below are some key concepts in this framework.

**BOAT**: [What is Boat?](readmes/what-is-boat.md)  
**CONFIG**: [A Bit About the Configs](readmes/config-design.md)

---

## 🏋️‍♂️ Training Models

Training scripts are available in the `train_scripts/` directory. To train a model:

```bash
python train_scripts/train_model.py --config configs/example_config.yaml
```

Replace `train_model.py` and `example_config.yaml` with your desired script and configuration file.

---

## 📊 Evaluating Models

Evaluation scripts are located in the `evaluate_scripts/` directory. To evaluate a trained model:

```bash
python evaluate_scripts/evaluate_model.py --config configs/example_config.yaml
```

Ensure that the configuration file points to the correct model checkpoint and evaluation dataset.

---

## 🛠️ Utilities

Common utility functions are housed in the `utils/` directory, including:

* Data loading and preprocessing
* Metric calculations
* Logging and checkpointing

These utilities promote code reuse and maintainability across different scripts.

---

## 📦 Packaging

The project includes packaging configurations:

* `setup.py`: Defines the package and its dependencies
* `pyproject.toml`: Specifies build system requirements

To build and install the package locally:

```bash
pip install .
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or suggestions, please open an issue or contact the repository owner directly.

---

Feel free to customize this `README.md` to better fit the specific details and requirements of your project.
