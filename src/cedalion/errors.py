"""Cedalion-specific exceptions."""


class CRSMismatchError(ValueError):
    """Error when coordinate reference systems do not match."""

    @classmethod
    def unexpected_crs(cls, expected_crs: str, found_crs: str):
        """Construct error for a coordinate system that differs from the expected one.

        Args:
            expected_crs: Name of the coordinate reference system that was required.
            found_crs: Name of the coordinate reference system that was encountered.

        Returns:
            CRSMismatchError: Ready-to-raise exception instance.
        """
        return cls(
            f"This operation expected coordinates to be in space "
            f"'{expected_crs}' but found them in '{found_crs}'."
        )

    @classmethod
    def wrong_transform(cls, current_crs: str, transform_crs: tuple[str]):
        """Construct error when an affine transform's CRS does not match the object's.

        Args:
            current_crs: CRS name of the object being transformed.
            transform_crs: CRS names encoded in the transform (source, target).

        Returns:
            CRSMismatchError: Ready-to-raise exception instance.
        """
        return cls(
            "The coordinate reference systems of this object "
            f"('{current_crs}') and of the transform ('{','.join(transform_crs)}') "
            "do not match."
        )
