import itertools
import mimetypes
import os
import os.path
from base64 import b64encode
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import ContentBlock, ImageContentBlock, VideoContentBlock, DataContentBlock, AudioContentBlock

def content_block_from_file(path=os.PathLike|str, mimetype: str|None=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' does not exist or cannot be accessed.")
    if mimetype is None:
        mimetype, _ = mimetypes.guess_type(path)
    if not mimetype:
        raise ValueError(f"Unable to determine mime-type for file '{path}'")

    media_type = mimetype.split('/')[0]
    with open(path, 'rb') as input_file:
        content = b64encode(input_file.read()).decode()

    return {
        "type": media_type,
        "mime_type": mimetype,
        "base64": content,
    }


@dataclass
class MultiModalResponse:
    blocks: "list[DataContentBlock]"

    @classmethod
    def from_bytes(cls, data: bytes|bytearray, mimetype: str):
        media_type = mimetype.split('/')[0]
        content = b64encode(data).decode()
        return cls(
            blocks=[
                {
                    "type": media_type,
                    "mime_type": mimetype,
                    "base64": content,
                }
            ]
        )

    @classmethod
    def from_file(cls, path: os.PathLike|str, mimetype: str|None=None):
        return cls(
            blocks=[
                content_block_from_file(path, mimetype)
            ]
        )

    @classmethod
    def from_files(cls, paths: list[os.PathLike|str], mimetypes: list[str]|None=None):
        if mimetypes is None:
            mimetypes = []
        blocks = [
            content_block_from_file(path, mimetype)
            for path, mimetype in itertools.zip_longest(paths, mimetypes, fillvalue=None)
        ]
        return cls(
            blocks=blocks
        )
