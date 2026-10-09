def check_specified_policy_kwargs(saved_policy_kwargs: dict, specified_policy_kwargs: dict) -> None:
    """
    Raise if a specified `policy_kwargs` entry differs from the saved one of a loaded model, since the architecture of
    a saved model cannot be changed. Only the specified entries are compared: unspecified ones keep their saved value.

    Args:
        saved_policy_kwargs: Policy parameters of the loaded model (may contain more entries, e.g., defaults).
        specified_policy_kwargs: `policy_kwargs` specified by the user.
    """
    not_set = object()
    differences = [
        f"`{key}`: stored {saved_policy_kwargs.get(key, 'not set')!r}, specified {value!r}"
        for key, value in sorted(specified_policy_kwargs.items())
        if saved_policy_kwargs.get(key, not_set) != value
    ]
    if differences:
        raise ValueError(
            "The specified policy_kwargs do not match the ones of the loaded model, whose architecture cannot be "
            f"changed: {'; '.join(differences)}"
        )
