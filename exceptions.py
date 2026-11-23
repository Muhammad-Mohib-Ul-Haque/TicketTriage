"""Custom exceptions for TicketTriage.

We raise these for problems we EXPECT (bad input, missing file, no internet).
main.py catches TicketTriageError and prints a friendly message instead of
crashing with a long Python traceback.
"""


class TicketTriageError(Exception):
    """Base error for the whole project. All our other errors inherit from it."""


class InvalidMessageError(TicketTriageError):
    """The customer message is empty, too short, too long, or has no letters."""


class APIUnavailableError(TicketTriageError):
    """The product API could not be reached and there was no saved copy to use."""
