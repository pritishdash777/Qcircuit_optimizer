import os
import torch


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path="models/gate_optimizer_cnn.pth"
):
    """
    Save the trained model weights.

    Example:
    models/gate_optimizer_cnn.pth
    """

    # Create the models folder if it doesn't exist
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    torch.save(
        model.state_dict(),
        path
    )

    print(
        f"Model saved to: {path}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model,
    path="models/gate_optimizer_cnn.pth"
):
    """
    Load previously trained model weights.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    state_dict = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"Model loaded from: {path}"
    )

    return model


# ============================================================
# CHECK IF MODEL EXISTS
# ============================================================

def model_exists(
    path="models/gate_optimizer_cnn.pth"
):
    """
    Check whether a saved model already exists.
    """

    return os.path.exists(
        path
    )