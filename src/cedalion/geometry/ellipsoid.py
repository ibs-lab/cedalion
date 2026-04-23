import numpy as np
import scipy.optimize
import cedalion.typing as cdt
import cedalion.dataclasses as cdc


@cdc.validate_schemas
def get_landmarks_for_headsize(
    circumference: cdt.QLength, nz_cz_iz: cdt.QLength, lpa_cz_rpa: cdt.QLength
) -> cdt.LabeledPoints:
    def ellipse_1020_costfunc(
        params: np.ndarray,
        circumference: float,
        nz_cz_iz: float,
        lpa_cz_rpa: float,
    ):
        a = params[0]  # R LPA to RPA axis
        b = params[1]  # A Nz to Iz axis
        c = params[2]  # S Cz axis

        # HC
        h = (a - b) ** 2 / (a + b) ** 2
        HC = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

        # IzNz
        h = (b - c) ** 2 / (b + c) ** 2
        IzNz = np.pi * (b + c) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))) * 1.2 / 2

        # AlAr
        h = (a - c) ** 2 / (a + c) ** 2
        AlAr = np.pi * (a + c) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))) * 1.2 / 2

        # [HC IzNz AlAr]
        return (
            (HC - circumference) ** 2
            + (IzNz - nz_cz_iz) ** 2
            + (AlAr - lpa_cz_rpa) ** 2
        )

    x0 = np.asarray([70, 70, 70])
    result = scipy.optimize.minimize(
        ellipse_1020_costfunc,
        x0,
        args=(
            circumference.to("mm").magnitude,
            nz_cz_iz.to("mm").magnitude,
            lpa_cz_rpa.to("mm").magnitude,
        ),
    )

    a, b, c = result.x

    r10p = 18 * np.pi / 180
    sin_r10p = np.sin(r10p)
    cos_r10p = np.cos(r10p)

    # fmt: off
    return cdc.build_labeled_points(
        [
            [          0,           0,           c],
            [-a*cos_r10p,           0, -c*sin_r10p],
            [ a*cos_r10p,           0, -c*sin_r10p],
            [          0,  b*cos_r10p, -c*sin_r10p],
            [          0, -b*cos_r10p, -c*sin_r10p],
        ],
        crs="ellipsoid",
        units = "mm",
        labels = ["Cz", "LPA", "RPA", "Nz", "Iz"],
        types = [cdc.PointType.LANDMARK] * 5
    )
    # fmt: on
