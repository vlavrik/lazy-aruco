"""
Lumache - Python library for cooks and food lovers.
"""

__version__ = "0.1.0"


class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]


def get_person(lastname, firstname, age, sex):
    """Gets a person identity.

    Parameters
    ----------
    lastname : str
        A lstname of a person
    firstname : str
        A firstname of a person
    age : int
        Age
    sex : bool
        Sex of a person. Defaults to True.

    Returns
    -------
    tuple
        lastname, firstname, age, sex

    Examples
    --------
    Fetching one of the persons.

    >>> get_person('Max', 'Muster', 23, True)
    """

    return lastname, firstname, age, sex
