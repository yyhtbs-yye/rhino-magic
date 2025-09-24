# Define the hook function
def named_forward_hook(layer_name, hook_memory, module, input, output):
    hook_memory[layer_name] = output
