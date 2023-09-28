import torch

def get_device():
    """
    Returns a device (cuda if available, cpu if not)

    Returns:
        torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze(x:any, max_depth:int=3, depth:int=0) -> None:
    """
    Prints an analysis of the provided x input.
    If x is an iterable, it recursively calls the function on it's children.

    Args:
        x (Any): Var to be analyzed.
        max_depth (int, optional): Max recursive depth. Defaults to 3.
        depth (int, optional): Current / Start depth. Defaults to 0.
    """
    assert max_depth < 10, "Too deep analysis requested!"
    if depth > max_depth:
        print()
        return
    print('\t' * depth, end='')
    
    if isinstance(x, torch.Tensor):
        print(f'{type(x)} \t shape: {x.shape}')
    elif hasattr(x, '__iter__') and not isinstance(x, str):
        print(f'{type(x)} \t len: {len(x)}')
        for i in x:
            analyze(i, depth+1, max_depth=max_depth)
    else:
        print(f'Type: {type(x)}')
