"""Cedalion-specific exceptions."""


class CRSMismatchError(ValueError):
    """Error when coordinate reference systems do not match."""

    @classmethod
    def unexpected_crs(cls, expected_crs: str, found_crs: str):
        return cls(
            f"This operation expected coordinates to be in space "
            f"'{expected_crs}' but found them in '{found_crs}'."
        )

    @classmethod
    def wrong_transform(cls, current_crs: str, transform_crs: tuple[str]):
        return cls(
            "The coordinate reference systems of this object "
            f"('{current_crs}') and of the transform ('{','.join(transform_crs)}') "
            "do not match."
        )
