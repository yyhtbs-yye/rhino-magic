## 🚢 What is a "Boat"?

In the `rhino-magic` framework, a **Boat** is a **self-contained container** that holds everything needed to train and evaluate a model. Think of it like a "model training ship" that carries:

* 🚀 The model(s)
* 🧮 The loss functions
* 🛠️ The optimizers
* 📈 The learning rate schedulers
* 📊 The evaluation metrics

This pattern helps separate the core *training logic* from the *trainer engine* itself. It makes the code modular, reusable, and easier to manage.

---

## 💡 Why Use a Boat?

Imagine training deep learning models as setting out on a journey. The **Trainer** is the captain, but the **Boat** is the vessel carrying all your essential gear.

* Boats define **how training happens** at a low level (forward, backward, step).
* Trainers define **when and in what order** things happen (epochs, logging, checkpointing).

This separation gives you flexibility: you can swap in different boats for different model architectures or tasks without changing your trainer code.

---

## 🧱 BaseBoat – The Blueprint

The `BaseBoat` is an abstract base class (like a template) that all boats inherit from. It provides:

| Function                         | Description                                     |
| -------------------------------- | ----------------------------------------------- |
| `models`, `losses`, `optimizers` | Hold your training components                   |
| `to(device)`                     | Move everything to GPU or CPU                   |
| `training_gradient_descent()`                | What happens in one training iteration          |
| `validation_step()`              | What happens in one validation step             |
| `build_optimizers()`         | How optimizers and schedulers are set up        |
| `save_state()` / `load_state()`  | Save/load the training state for resuming       |
| `manual_backward()`              | Support for manual or distributed backward pass |

You must implement the key training/validation methods in any subclass.

---

## 🧠 BaseDiffusionBoat – A Practical Example

The `BaseDiffusionBoat` is a real boat built for training diffusion models. It uses the BaseBoat interface and adds features like:

* 🛠 `build_module()` to dynamically create models from configs
* ✅ Loss and metric setup via `build_losses()` and `build_metrics()`
* 🔁 EMA (Exponential Moving Average) to smooth model weights during training
* 🖼️ Visual logging during validation

By using `BaseDiffusionBoat`, developers don’t need to worry about the detailed training steps—just plug in the right config and go.

---

## 🧩 Key Design Pattern Benefits

| Benefit           | Explanation                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Modularity**    | You can create different boats for different models or tasks without rewriting the trainer.                           |
| **Reusability**   | Common logic (like `.to(device)` or saving/loading) lives in `BaseBoat` and gets reused everywhere.                   |
| **Extensibility** | Add new behaviors (e.g., metrics, EMA, visualization) in subclasses without touching the trainer.                     |
| **Decoupling**    | Trainers manage training *flow*, while boats manage training *logic*. This makes each part easier to test and modify. |

---

## 🧭 Summary

> A **Boat** is a self-contained package for training logic, built to be plugged into a **Trainer**. It handles what models to use, how to train them, and how to evaluate them—all while staying modular and easy to extend.
