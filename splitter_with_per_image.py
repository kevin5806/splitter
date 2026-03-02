from src.splitter_with_per_image import *  # noqa: F401,F403
from src import splitter_with_per_image as _impl

_index_to_alpha = _impl._index_to_alpha
_generate_crop_codes = _impl._generate_crop_codes
_crop_with_white_padding = _impl._crop_with_white_padding


if __name__ == "__main__":
    main()
