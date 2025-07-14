import torch

def average_gradients_into_main(boats, devices):
    """
    In-place average of replica gradients into replica-0.

    After this call, only replica-0's parameters have the averaged grads.
    """
    n_devices = len(devices)
    
    # Get all model keys from the first replica
    model_keys = boats[0].models.keys()
    
    # Average gradients for each model
    for model_key in model_keys:
        if not hasattr(boats[0].models[model_key], 'parameters'):
            continue

        # Walk through params position-wise across replicas for this model
        for params in zip(*(m.models[model_key].parameters() for m in boats)):
            main_grad = params[0].grad
            if main_grad is None:
                continue 

            # Sum grads from the other replicas into main_grad
            for p in params[1:]:
                if p.grad is not None:
                    main_grad.data.add_(p.grad.to(main_grad.device))

            # Average
            main_grad.data.div_(n_devices)
