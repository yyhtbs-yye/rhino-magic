# 🦏 Rhino-Magic

**Rhino-Magic** is a modular and extensible framework for low-level vision research – specifically focusing on image generation, image restoration, and even video generation – designed for rapid experimentation. It provides a structured approach to training and evaluating models, leveraging YAML configuration files and reusable components to streamline workflows. The design draws inspiration from libraries like PyTorch Lightning (though it’s a custom framework, not Lightning itself) to separate the concerns of *what* the model does from *how* the training loop is run.

## ⭐ Features

* **Low-Level Vision Focus:** Provides tools for tasks such as image synthesis, image enhancement/restoration, and (planned) video generation, making it ideal for research in these domains.
* **Modular "Boat" Architecture:** Encapsulates all training logic (models, losses, optimizers, schedulers, metrics) into self-contained modules called *Boats*, separate from the training loop, for maximum flexibility and reuse. Swap in different Boats for different tasks without changing the Trainer code.
* **Flexible Configuration System:** Uses YAML configuration files to define experiments. Every component (datasets, models, optimizers, etc.) is specified by a `path` and `name` for dynamic importing, with distinct sections for each concern (model/boat, optimization, data, training parameters, etc.). Configs support variables and inheritance (*import*) to avoid repetition and promote reuse.
* **Extensible and Reusable:** The decoupled design (Trainer vs. Boat vs. Config) makes it easy to extend the framework. You can add new models, metrics, or training strategies by creating new Boat subclasses or config entries **without** modifying the core training engine. Common functionalities (moving to device, saving state, etc.) are provided in base classes so they don’t have to be rewritten for each new model.
* **Built on PyTorch & Diffusers:** Leverages PyTorch for model training and integrates with HuggingFace Diffusers for state-of-the-art diffusion models. For example, the framework includes a `BaseDiffusionBoat` tailored for diffusion models, with features like automatic module building from configs, EMA (Exponential Moving Average) of model weights, and visual result logging during validation. This makes it straightforward to experiment with diffusion-based image generation and will support similar techniques for video.

## 📁 Project Structure

```
rhino-magic/
├── configs/              # YAML configuration files for experiments
├── evaluate_scripts/     # Scripts for model evaluation
├── jupyter_notebooks/    # Notebooks for data exploration and prototyping
├── readmes/              # Additional documentation (linked from this README)
├── rhino/                # Core library modules (Boat implementations, models, etc.)
├── train_scripts/        # Scripts for training models
├── trainer/              # Training loop logic (Trainer classes and utilities)
├── utils/                # Helper functions and utilities
├── work_dirs/            # Output directories for experiment results (empty by default)
├── pyproject.toml        # Build system configuration
├── requirements.txt      # Python dependencies
└── setup.py              # Package installation script
```

This structure separates configuration, training logic, and core modules for clarity. Notably, the **`rhino/`** package contains the core classes (like Boats and model definitions), while **`trainer/`** contains the Trainer classes that orchestrate the training procedure.

## 🚀 Getting Started

### Prerequisites

* Python 3.11
* PyTorch 2.2.1 or higher (for model training)
* HuggingFace Diffusers 0.33.1 (for diffusion models)
* NumPy 1.24.4 or higher
* *(Recommended:* Create a Python virtual environment for installation\*)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yyhtbs-yye/rhino-magic.git
   cd rhino-magic
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Rhino-Magic (optional):** If you want to use it as a package, install in editable mode:

   ```bash
   pip install -e .
   ```

   This will allow you to import `rhino` and `trainer` modules in your own scripts.

Now you’re ready to use the framework in your projects or experiments.

## ⚙️ Configuration

Experiment settings are managed through YAML files located in the `configs/` directory. These config files define all aspects of training: model architecture, training hyperparameters, dataset paths, optimizer/scheduler settings, etc. Rather than hard-coding these in Python, you edit or create a config YAML to set up an experiment.

Key aspects of the configuration system:

* **Dynamic Components:** Each component is specified by a Python module `path` and the class `name` to instantiate. This tells Rhino-Magic what class to import and initialize for that part of the pipeline. For example, a model might be defined with:

  ```yaml
  model:
    path: models.unet            # module path
    name: UNet2DModel            # class name
    ...
  ```

  Similar patterns apply to datasets, loss functions, optimizers, schedulers, Boats, etc.

* **Organized Sections:** The config is divided into sections for each concern (e.g., `boat` for the training container logic, `optimization` for optimizer and learning rate schedule, `trainer` for training loop settings like epochs/devices, `data` for dataset and dataloaders, `validation` for evaluation metrics and frequency, as well as optional `logging` or `callbacks`). This clear separation makes configs easy to navigate.

* **Variables and Imports:** Configs support *placeholders* (like `$batch_size`, defined in a `_vars` section) that get replaced throughout the file. They also support inheriting from base configs using an `_import` directive. For example, you might have a base config with common settings and then import it in your specific experiment config to avoid duplication. This makes it simple to create variations of experiments by overriding only certain fields.

Using this config system, you can quickly tweak architectures, datasets, or hyperparameters by editing YAML, without changing any code. An example configuration file is provided (e.g., `configs/example_config.yaml`) to illustrate the format.

## ⚓ Key Design Concepts

### The "Boat" and Trainer Paradigm

Rhino-Magic introduces the concept of a **Boat** as a self-contained training module. A Boat carries everything needed for a training voyage: the model(s), loss functions, optimizers, schedulers, and evaluation metrics. The **Trainer**, on the other hand, is like the captain that steers the Boat – it manages the overall training loop (epochs, batching, logging, checkpointing), calling on the Boat to perform the low-level tasks for each step.

In practice, `BaseBoat` is an abstract base class that defines the interface and common utilities for all boats. It includes methods such as `training_step()`, `validation_step()`, `configure_optimizers()`, `to(device)`, and state save/load functions, which outline **what happens** during a training iteration or validation pass. Specific boat classes (for example, a `DiffusionBoat` for diffusion models) inherit from BaseBoat and implement these methods with task-specific logic. This design cleanly separates **training logic** (inside the Boat) from the **training flow control** (handled by the Trainer).

**Why "Boat"?** This metaphor emphasizes modularity. You can easily swap in a different Boat to train a different model or perform a different task, without needing to rewrite the entire training loop. For instance, today you might use a diffusion model Boat for image generation, and tomorrow a different Boat for video generation – the Trainer code remains the same. This decoupling also makes the system more maintainable and extensible, since improvements to the BaseBoat (like better checkpointing or distributed training support) automatically benefit all Boat implementations.

One provided implementation is the **`BaseDiffusionBoat`**, which extends BaseBoat for training diffusion models. It adds convenient features like:

* Dynamic model construction from the config (`build_module()` to create models defined in YAML)
* Automated loss and metric setup (`configure_losses()`, `configure_metrics()`)
* **EMA** (Exponential Moving Average) of model weights during training for more stable long-term training
* Built-in visualization hooks to log generated images during validation steps (useful for tracking generative model progress)

By using the Boat/Trainer pattern, Rhino-Magic enables a Lightning-like experience where you focus on defining *what* your model and training logic is, and the framework handles *when* and *how* to run it. This results in a highly flexible training pipeline that can accommodate new research ideas with minimal changes to the codebase.

### Config-Driven Workflows

The YAML-based configuration system is a core pillar of Rhino-Magic’s design. It acts as a blueprint for experiments. Instead of writing new Python scripts for each experiment, you configure the experiment in a declarative way. This offers several benefits:

* **Separation of Concerns:** Your code (Boats, Trainers, models) implements logic, while the configs specify parameters and choices. This keeps code clean and generic, and all experiment-specific settings in one place.
* **Reusability:** You can easily swap out components (e.g., try a different optimizer or model architecture) by editing the config, without touching the Python code.
* **Flexibility:** Need a new training setup? Copy an existing YAML config, change a few fields, and you’re ready to go. This lowers the barrier to trying out new ideas or hyperparameters.
* **Transparency and Reproducibility:** The full training recipe is captured in one file. It’s easy to track exactly what settings were used for a run, and to share that with others or in papers. You can also version control your configs to keep a history of experiments.
* **Scalability:** As your project grows, you can organize configs hierarchically (base configs, specialized configs) and use imports to avoid duplication. New components (say, a new scheduler or metric) can be integrated by adding a small config snippet and implementing the class – no need to alter the entire framework.

In summary, Rhino-Magic’s config-driven approach and Boat architecture work hand-in-hand to enable a **highly modular, extensible, and researcher-friendly** workflow. You spend more time on core ideas and less on writing boilerplate training loops.

## 🏋️‍♂️ Training Models

Training scripts are provided in the `train_scripts/` directory. For example, after configuring your experiment YAML, you can kick off training with:

```bash
python train_scripts/train_model.py --config configs/example_config.yaml
```

Replace `train_model.py` with the training script of your choice, and `configs/example_config.yaml` with your config file. This will launch the Trainer which reads your config, builds the necessary modules (model, Boat, etc.), and begins training.

During training, checkpoints and logs will be saved under `work_dirs/` (or a path specified in your config). You can monitor training progress via console logs or any logging callbacks you configured (e.g., TensorBoard, if integrated).

*Developer note:* You can create new training scripts or modify existing ones for custom workflows. For most use cases, the provided generic scripts should suffice, reading all needed settings from the config.

## 📊 Evaluating Models

Evaluation scripts live in the `evaluate_scripts/` directory. After training a model, you can evaluate its performance using a similar interface. For example:

```bash
python evaluate_scripts/evaluate_model.py --config configs/example_config.yaml
```

Ensure that the config file points to the correct model checkpoint (either in the config’s `trainer` section or via a command-line override) and the desired evaluation dataset or metrics. The evaluation script will load the trained model (using your Boat/Trainer setup) and compute the metrics or generate output as specified.

You can create specialized evaluation scripts if needed (for example, for custom metrics or visualizations), but the provided ones are a good starting point for typical use cases.

## 🛠️ Utilities

Common utility functions are available in the `utils/` directory for tasks like data loading, image preprocessing, metric calculations, logging, and checkpointing. These utilities are used across training and evaluation scripts to avoid code duplication. If you develop new functionality, consider adding a reusable helper in `utils/` if it might be used elsewhere.

Some examples of utilities you might find or add:

* Custom dataset classes or data augmentations (beyond basic ones)
* Image or video processing helpers (cropping, normalization, conversion between color spaces, etc.)
* Logging hooks or callbacks for advanced monitoring
* Checkpoint management (saving best model, etc.)

Leveraging the utilities will help keep your Boat and Trainer code focused on the core logic, delegating ancillary tasks to well-tested helper functions.

## 📦 Packaging

Rhino-Magic is set up as a Python package for convenience. The `setup.py` and `pyproject.toml` define how the project can be installed and what its dependencies are. You can install the package locally (as shown above) to integrate it with other projects or to easily run scripts from anywhere.

Key notes on packaging:

* After installing with `pip install -e .`, you can `import rhino` and `import trainer` in Python, which correspond to the `rhino/` and `trainer/` modules.
* This project currently has minimal installation requirements (mostly relying on PyTorch and related libraries which you install separately). If you add new dependencies for your experiments, make sure to update `requirements.txt` or the setup configuration.

## 📄 License

This project is licensed under the **MIT License**. By using this code, you agree to the terms of the MIT license, which permits use, modification, and distribution of the code as long as the license notice is preserved. See the **[LICENSE](LICENSE)** file for the full license text.

## 📬 Contact & Support

For questions, issues, or suggestions, please open an issue on the GitHub repository. You can also contact the repository owner via email at **[yyhtbs0@gmail.com](mailto:yyhtbs0@gmail.com)** for direct inquiries. Contributions are welcome – feel free to submit pull requests for new features or fixes. We hope **Rhino-Magic** accelerates your research and development in low-level vision and generative modeling!
