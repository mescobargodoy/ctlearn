from importlib.resources import files, as_file

from ctapipe.core import Provenance
from ctapipe.core.traits import TraitError
from ctapipe.instrument import SubarrayDescription
from ctapipe.instrument.optics import FocalLengthKind

import tensorflow as tf
from tensorflow.keras.losses import Loss

__all__ = ["get_lst1_subarray_description", "validate_trait_dict", "RootMeanSquaredErrorLoss"]

def get_lst1_subarray_description(focal_length_choice=FocalLengthKind.EFFECTIVE):
    """
    Load subarray description from bundled file
    
    Parameters
    ----------
    focal_length_choice : FocalLengthKind
        Choice of focal length to use.  Options are ``FocalLengthKind.EQUIVALENT``
        and ``FocalLengthKind.EFFECTIVE``. Default is ``FocalLengthKind.EFFECTIVE``.

    Returns
    -------
    SubarrayDescription
        Subarray description of the LST-1 telescope.
    """
    with as_file(files("ctlearn") / "resources/LST-1_SubarrayDescription.h5") as path:
        Provenance().add_input_file(path, role="SubarrayDescription")
        return SubarrayDescription.from_hdf(path, focal_length_choice=focal_length_choice)

def validate_trait_dict(dict, required_keys):
    """
    Validate that a dictionary contains all required keys.

    Parameters
    ----------
    dict : dict
        Dictionary to validate.
    required_keys : set
        Set of required keys.

    Returns
    -------
    bool
        True if the dictionary contains all required keys.  Otherwise, raises a TraitError.
    """
    missing_keys = required_keys - dict.keys()
    if missing_keys:
        raise TraitError(f"Dict is missing required key(s): {', '.join(missing_keys)}")
    return True

class RootMeanSquaredErrorLoss(Loss):
    def __init__(self, name="root_mean_squared_error_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def get_config(self):
        config = super().get_config()
        return config