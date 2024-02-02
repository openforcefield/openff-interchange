import uuid
from io import StringIO

from openff.toolkit.utils._viz import TopologyNGLViewStructure

from openff.interchange import Interchange


class InterchangeNGLViewStructure(TopologyNGLViewStructure):
    """Subclass of the toolkit's NGLView interface."""

    def __init__(
        self,
        interchange: Interchange,
        ext: str = "PDB",
    ):
        self.interchange = interchange
        self.ext = ext.lower()
        self.params: dict = dict()
        self.id = str(uuid.uuid4())

    def get_structure_string(self) -> str:
        """Get structure as a string."""
        with StringIO() as f:
            self.interchange.to_pdb(f, include_virtual_sites=True)
            structure_string = f.getvalue()
        return structure_string
