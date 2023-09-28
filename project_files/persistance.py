# For classes and methods dealing with persisting items.
# Torch -> save torch models
# Pickle for everything else.

import torch
import pickle
from typing import Any, Dict, Optional

# region torch model persistance
def save_model(model) -> None:
    """
    Save a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
    """
    torch.save(model.state_dict(), self.file_path)

def load_model(model, device=torch.device('cpu')) -> None:
    """
    Load a PyTorch model from a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be loaded into.
        device (torch.device, optional): The device on which the model should be loaded.
            Defaults to torch.device('cpu').
    """
    model.load_state_dict(torch.load(self.file_path, map_location=device))
    model.to(device)
    model.eval()

# endregion

#region Pickle
class PickleHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def create(self, key: str, value: Any) -> None:
        """
        Create a new entry in the pickle file.

        Args:
            key (str): The key for the entry.
            value (Any): The value associated with the key.
        """
        data = self._load_data()
        data[key] = value
        self._save_data(data)

    def read(self, key: str) -> Optional[Any]:
        """
        Read the value associated with a given key.

        Args:
            key (str): The key for which to retrieve the value.

        Returns:
            Any: The value associated with the key, or None if key is not found.
        """
        data = self._load_data()
        return data.get(key)

    def update(self, key: str, value: Any) -> None:
        """
        Update the value associated with a given key.

        Args:
            key (str): The key for which to update the value.
            value (Any): The new value.
        """
        data = self._load_data()
        if key in data:
            data[key] = value
            self._save_data(data)
        else:
            raise KeyError(f"Key '{key}' not found.")

    def delete(self, key: str) -> None:
        """
        Delete the entry associated with a given key.\n
        Returns success.

        Args:
            key (str): The key to be deleted.
        """
        data = self._load_data()
        if key in data:
            del data[key]
            self._save_data(data)
            return True
        else:
            return False

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return {}
        except pickle.UnpicklingError:
            # If there is a file corruption type error:
            return {}

    def _save_data(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, 'wb') as file:
            pickle.dump(data, file)
#endregion

