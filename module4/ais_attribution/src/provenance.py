import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def create_provenance_record(
    ais_data_source,
    input_file,
    total_records,
    filtered_records=None,
    normalized_records=None
):
    """
    Create provenance information for AIS processing.
    """

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "ais_data_source": ais_data_source,

        "input_file": str(input_file),

        "total_records": int(
            total_records
        ),

        "normalized_records": (
            int(normalized_records)
            if normalized_records is not None
            else None
        ),

        "filtered_records": (
            int(filtered_records)
            if filtered_records is not None
            else None
        ),

        "coordinate_reference_system": (
            "EPSG:4326"
        ),

        "processing_notes": [
            "AIS records were normalized "
            "to the internal schema.",

            "Invalid records were skipped "
            "during cleaning.",

            "AIS gaps are informational only "
            "and do not reduce candidate ranking.",

            "AIS completeness is contextual only "
            "and is not included in the composite score.",

            "Composite scores are heuristic "
            "ranking signals and not probabilities."
        ]
    }

    return record


def save_provenance(
    provenance_record,
    output_path="outputs/provenance.json"
):
    """
    Save provenance information as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            provenance_record,
            file,
            indent=4
        )

    logger.info(
        f"Provenance saved to: {output_path}"
    )

    return output_path


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    provenance = create_provenance_record(
        ais_data_source="synthetic",
        input_file=(
            "data/synthetic/"
            "synthetic_ais.csv"
        ),
        total_records=600,
        normalized_records=600,
        filtered_records=60
    )

    output_path = save_provenance(
        provenance
    )

    print(
        "\nProvenance logging completed."
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"AIS source: "
        f"{provenance['ais_data_source']}"
    )

    print(
        f"Total records: "
        f"{provenance['total_records']}"
    )

    print(
        f"Filtered records: "
        f"{provenance['filtered_records']}"
    )