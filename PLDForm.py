"""Backward-compatible import wrapper for PLD form classes.

New code should import from ``pld_workflow.form`` instead:

    from pld_workflow.form import GenerateForm
"""

from pld_workflow.form import GenerateForm, MessageWindow

__all__ = ["GenerateForm", "MessageWindow"]
