"""GUI Sections package for Project QuickNav."""

from .project_input import ProjectInputSection
from .navigation_mode import NavigationModeSection
from .folder_mode import FolderModeSection
from .document_mode import DocumentModeSection
from .options import OptionsSection
from .ai_assistant import AIAssistantSection
from .status import StatusSection
from .action_buttons import ActionButtonsSection

__all__ = [
    'ProjectInputSection',
    'NavigationModeSection',
    'FolderModeSection',
    'DocumentModeSection',
    'OptionsSection',
    'AIAssistantSection',
    'StatusSection',
    'ActionButtonsSection',
]
