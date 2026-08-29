from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
EARTH_RADIUS_MILES = 3958.8


# ---------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------
def calculate_haversine_distance(
    pickup_lat: pd.Series,
    pickup_lon: pd.Series,
    delivery_lat: pd.Series,
    delivery_lon: pd.Series,
) -> pd.Series:
    """
    Calculate straight-line geographic distance in miles
    between pickup and delivery coordinates.

    Parameters
    ----------
    pickup_lat, pickup_lon, delivery_lat, delivery_lon
        Latitude and longitude values in decimal degrees.

    Returns
    -------
    pd.Series
        Haversine distance in miles.
    """

    lat1 = np.radians(
        pickup_lat.astype(float)
    )
    lon1 = np.radians(
        pickup_lon.astype(float)
    )

    lat2 = np.radians(
        delivery_lat.astype(float)
    )
    lon2 = np.radians(
        delivery_lon.astype(float)
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2.0) ** 2
    )

    # Numerical safety
    a = np.clip(
        a,
        0.0,
        1.0,
    )

    distance = (
        2
        * EARTH_RADIUS_MILES
        * np.arcsin(
            np.sqrt(a)
        )
    )

    return pd.Series(
        distance,
        index=pickup_lat.index,
    )


# ---------------------------------------------------------
# Route bearing
# ---------------------------------------------------------
def calculate_route_bearing(
    pickup_lat: pd.Series,
    pickup_lon: pd.Series,
    delivery_lat: pd.Series,
    delivery_lon: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Represent approximate route direction using sine and
    cosine of the initial geographic bearing.

    Using sine/cosine avoids a discontinuity between
    directions close to 0 and 360 degrees.
    """

    lat1 = np.radians(
        pickup_lat.astype(float)
    )
    lon1 = np.radians(
        pickup_lon.astype(float)
    )

    lat2 = np.radians(
        delivery_lat.astype(float)
    )
    lon2 = np.radians(
        delivery_lon.astype(float)
    )

    delta_lon = lon2 - lon1

    y = (
        np.sin(delta_lon)
        * np.cos(lat2)
    )

    x = (
        np.cos(lat1)
        * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(delta_lon)
    )

    bearing = np.arctan2(
        y,
        x,
    )

    bearing_sin = pd.Series(
        np.sin(bearing),
        index=pickup_lat.index,
    )

    bearing_cos = pd.Series(
        np.cos(bearing),
        index=pickup_lat.index,
    )

    return (
        bearing_sin,
        bearing_cos,
    )


# ---------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------
def create_features(
    df: pd.DataFrame,
    include_city_categories: bool = False,
    include_market_signals: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create modeling features from preprocessed freight data.

    Parameters
    ----------
    df:
        DataFrame after apply_preprocessing().

    include_city_categories:
        If True, retain pickup and delivery city names as
        categorical model inputs.

        Default is False because the final validation data
        contains cities not observed in the labeled data.
        Geographic coordinates provide a more robust
        representation for those unseen cities.

    include_market_signals:
        If True, include market_index_clean, its missing
        flag, and quote_signal.

        This option allows later feature-ablation testing
        to determine whether the supplied market signals
        improve chronological validation performance.

    Returns
    -------
    X:
        Model feature DataFrame.

    categorical_features:
        Names of categorical columns retained in X.
    """

    result = df.copy()

    # -----------------------------------------------------
    # Required-column checks
    # -----------------------------------------------------
    required_columns = [
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight_clean",
        "weight_missing_flag",
        "weight_negative_flag",
        "date",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required preprocessed columns: "
            + ", ".join(missing_required)
        )

    # -----------------------------------------------------
    # 1. Date features
    # -----------------------------------------------------
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        raise ValueError(
            "Invalid date values detected during "
            "feature engineering."
        )

    result["year"] = (
        result["date"].dt.year
    )

    result["month"] = (
        result["date"].dt.month
    )

    result["day_of_month"] = (
        result["date"].dt.day
    )

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_year"] = (
        result["date"].dt.dayofyear
    )

    result["week_of_year"] = (
        result["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    result["is_weekend"] = (
        result["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    result["is_month_start"] = (
        result["date"]
        .dt.is_month_start
        .astype(int)
    )

    result["is_month_end"] = (
        result["date"]
        .dt.is_month_end
        .astype(int)
    )

    # -----------------------------------------------------
    # 2. Cyclical time features
    # -----------------------------------------------------
    # Day of week repeats every seven days.
    result["day_of_week_sin"] = np.sin(
        2
        * np.pi
        * result["day_of_week"]
        / 7
    )

    result["day_of_week_cos"] = np.cos(
        2
        * np.pi
        * result["day_of_week"]
        / 7
    )

    # Month repeats every twelve months.
    result["month_sin"] = np.sin(
        2
        * np.pi
        * result["month"]
        / 12
    )

    result["month_cos"] = np.cos(
        2
        * np.pi
        * result["month"]
        / 12
    )

    # Day of year captures annual seasonality.
    result["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    result["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    # -----------------------------------------------------
    # 3. Distance features
    # -----------------------------------------------------
    result["log_distance"] = np.log1p(
        result["distance"]
    )

    # -----------------------------------------------------
    # 4. Geographic differences
    # -----------------------------------------------------
    result["latitude_change"] = (
        result["delivery_lat"]
        - result["pickup_lat"]
    )

    result["longitude_change"] = (
        result["delivery_lon"]
        - result["pickup_lon"]
    )

    result["absolute_latitude_change"] = (
        result["latitude_change"].abs()
    )

    result["absolute_longitude_change"] = (
        result["longitude_change"].abs()
    )

    # Route midpoint
    result["route_mid_lat"] = (
        result["pickup_lat"]
        + result["delivery_lat"]
    ) / 2.0

    result["route_mid_lon"] = (
        result["pickup_lon"]
        + result["delivery_lon"]
    ) / 2.0

    # -----------------------------------------------------
    # 5. Straight-line geographic distance
    # -----------------------------------------------------
    result["haversine_distance"] = (
        calculate_haversine_distance(
            result["pickup_lat"],
            result["pickup_lon"],
            result["delivery_lat"],
            result["delivery_lon"],
        )
    )

    # -----------------------------------------------------
    # 6. Detour ratio
    # -----------------------------------------------------
    # Compares reported route mileage with geographic
    # straight-line distance.
    result["detour_ratio"] = (
        result["distance"]
        / (
            result["haversine_distance"]
            + 1.0
        )
    )

    # -----------------------------------------------------
    # 7. Route direction
    # -----------------------------------------------------
    (
        result["bearing_sin"],
        result["bearing_cos"],
    ) = calculate_route_bearing(
        result["pickup_lat"],
        result["pickup_lon"],
        result["delivery_lat"],
        result["delivery_lon"],
    )

    # -----------------------------------------------------
    # 8. Load interaction features
    # -----------------------------------------------------
    result["distance_x_weight"] = (
        result["distance"]
        * result["weight_clean"]
        / 10000.0
    )

    result["weight_per_1000_miles"] = (
        result["weight_clean"]
        / (
            result["distance"] + 1.0
        )
    )

    # -----------------------------------------------------
    # 9. Select model features
    # -----------------------------------------------------
    feature_columns = [
        # Primary numeric load information
        "distance",
        "log_distance",
        "weight_clean",
        "weight_missing_flag",
        "weight_negative_flag",

        # Equipment
        "equipment",

        # Original coordinates
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",

        # Geographic derivatives
        "latitude_change",
        "longitude_change",
        "absolute_latitude_change",
        "absolute_longitude_change",
        "route_mid_lat",
        "route_mid_lon",
        "haversine_distance",
        "detour_ratio",
        "bearing_sin",
        "bearing_cos",

        # Calendar features
        "year",
        "month",
        "day_of_month",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",

        # Cyclical calendar features
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",

        # Interactions
        "distance_x_weight",
        "weight_per_1000_miles",
    ]

    categorical_features = [
        "equipment",
    ]

    # -----------------------------------------------------
    # 10. Optional city categories
    # -----------------------------------------------------
    if include_city_categories:
        feature_columns.extend(
            [
                "pickup",
                "delivery",
            ]
        )

        categorical_features.extend(
            [
                "pickup",
                "delivery",
            ]
        )

    # -----------------------------------------------------
    # 11. Optional supplied market features
    # -----------------------------------------------------
    if include_market_signals:

        market_required = [
            "market_index_clean",
            "market_index_missing_flag",
            "quote_signal",
        ]

        missing_market = [
            column
            for column in market_required
            if column not in result.columns
        ]

        if missing_market:
            raise ValueError(
                "Missing market feature columns: "
                + ", ".join(missing_market)
            )

        feature_columns.extend(
            [
                "market_index_clean",
                "market_index_missing_flag",
                "quote_signal",
            ]
        )

    # -----------------------------------------------------
    # 12. Final feature DataFrame
    # -----------------------------------------------------
    X = result[
        feature_columns
    ].copy()

    # CatBoost categorical values should be strings.
    for column in categorical_features:
        X[column] = (
            X[column]
            .astype(str)
        )

    # -----------------------------------------------------
    # 13. Final numerical validity checks
    # -----------------------------------------------------
    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    if X[numeric_columns].isna().any().any():
        problem_columns = (
            X[numeric_columns]
            .columns[
                X[numeric_columns]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Feature engineering produced missing values "
            "in: "
            + ", ".join(problem_columns)
        )

    numeric_array = (
        X[numeric_columns]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise ValueError(
            "Feature engineering produced "
            "non-finite numerical values."
        )

    return (
        X,
        categorical_features,
    )