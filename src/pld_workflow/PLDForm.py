"""Compatibility module for legacy imports.

Prefer:

    from pld_workflow.form import GenerateForm
"""

from .form import GenerateForm, MessageWindow

__all__ = ["GenerateForm", "MessageWindow"]
