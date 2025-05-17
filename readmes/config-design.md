## 🧾 What Are These Configs?

The configuration system in `rhino-magic` is like a **mission control dashboard** for setting up training runs.

Instead of hardcoding models, training logic, data, and hyperparameters in Python files, everything is defined in **YAML configuration files**. These act as a **blueprint** to dynamically build and wire up components at runtime.

---

## 🧩 Config Design Pattern

The pattern follows a few core ideas:

### 1. **Dynamic Importing with `path` + `name`**

Every component is defined using:

```yaml
path: 'module.path'
name: 'ClassName'
```

This tells the system:

> “Import this class and build it using the config below.”

For example, this applies to:

* Models (`UNet2DModel`, `AutoencoderKLWrapper`)
* Schedulers (`DDIMScheduler`)
* Loss functions (`WeightedMSELoss`)
* Optimizers (`Adam`)
* Datasets (`BasicImageDataset`)
* Trainers, Callbacks, Boats—even metrics

### 2. **Component-Based Config Sections**

Each major part of training has its own clearly scoped section:

* `boat`: How to construct your training container (model, solver, loss, scheduler)
* `optimization`: How to train it (optimizer, EMA, LR scheduler)
* `trainer`: Controls the training loop (epochs, devices, evaluation)
* `data`: What to train on (dataset, preprocessing)
* `validation`: How to evaluate (metrics, visualization)
* `logging`, `callbacks`: Optional extras (logging frequency, checkpointing)

### 3. **Template-Like Variables**

You’ll see placeholders like `$in_channels`, `$batch_size`. These are defined at the bottom (`_vars`) or in imported files (`_import`) and get **filled in** before the config is parsed.

This avoids repetition and makes it easy to run experiments with different settings.

### 4. **Composable and Overridable**

Configs are hierarchical and can `import` base configs to override or extend them:

```yaml
_import: configs/base/train_base_ddim.yaml
```

This allows:

* Sharing base setups (e.g., training on DDIM)
* Specializing with minimal changes (e.g., change model or dataset)

---

## 🧠 Why Use This Pattern?

| Benefit                    | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| **Separation of concerns** | Code stays clean—logic in Python, parameters in YAML                    |
| **Reusability**            | Swap out models, losses, datasets without rewriting training code       |
| **Flexibility**            | Create new training setups quickly by copying/modifying configs         |
| **Transparency**           | Easy to inspect and audit exactly how a model was trained               |
| **Scalability**            | Add new components (e.g. a new scheduler or metric) with minimal effort |

---

## 📦 Summary

> The config system is a plug-and-play blueprint for experiments. It defines *what* gets built, *how* it's wired together, and *what settings* are used—without writing new code.

This pattern makes `rhino-magic` highly modular, adaptable, and ideal for research or large-scale experimentation.
