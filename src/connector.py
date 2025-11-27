from __future__ import annotations

import logging
import os
import time
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
from typing import Iterator, Optional

import requests
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError, RequestException

import aiod
from aiod.authentication import set_token, Token
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10
MAX_DESCRIPTION_LENGTH = 1800

PLATFORM_NAME = "zenodo"

STOP_ON_UNEXPECTED_ERROR: bool = False
PER_DATASET_DELAY: Optional[float] = None
ZENODO_PAGE_SIZE: int = 25


class ParsingError(Exception):
    pass


class ServerError(Exception):
    pass


def _paginate_zenodo_records(page_size: Optional[int] = None) -> Iterator[dict]:
    base_url = "https://zenodo.org/api/records"
    page = 1

    if page_size is None:
        page_size = ZENODO_PAGE_SIZE

    while True:
        params = {
            "page": page,
            "size": page_size,
            "all_versions": 1,
            "sort": "mostrecent",
        }
        logger.debug("Requesting Zenodo records: %s params=%s", base_url, params)

        try:
            response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        except RequestException as e:
            logger.warning(
                "Request error while fetching %s (page %s): %s; skipping this page and continuing.",
                base_url,
                page,
                e,
            )
            page += 1
            continue

        if not response.ok:
            try:
                content = response.json()
            except Exception:
                content = response.text
            logger.warning(
                "Non-OK response while fetching %s (page %s): (%s) %s; skipping this page and continuing.",
                response.url,
                page,
                response.status_code,
                content,
            )
            page += 1
            continue

        try:
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.exception("Error parsing Zenodo response")
            raise ParsingError(
                f"Could not parse Zenodo response ({response.status_code}): "
                f"{response.content}"
            ) from e

        if not hits:
            logger.debug("No more Zenodo records returned, stopping pagination.")
            break

        logger.debug("Fetched %d Zenodo records (page %d)", len(hits), page)
        for record in hits:
            yield record

        page += 1


def list_records(from_id: Optional[int] = None) -> Iterator[dict]:
    from_id = from_id or 0
    for record in _paginate_zenodo_records():
        try:
            identifier = int(record["id"])
        except Exception:
            logger.error("Zenodo record without integer 'id': %s", record, exc_info=True)
            continue

        if identifier < from_id:
            continue

        try:
            full_record = fetch_zenodo_record(identifier)
        except (ServerError, ParsingError, RequestException) as e:
            logger.warning(
                "Skipping Zenodo record %s due to fetch error: %s", identifier, e
            )
            continue

        yield full_record


def fetch_zenodo_record(identifier: int) -> dict:
    url = f"https://zenodo.org/api/records/{identifier}"
    logger.debug("Fetching Zenodo record %s", url)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except RequestException as e:
        logger.warning("Request error while fetching %s: %s", url, e)
        raise ServerError(
            f"Error while fetching {url} from Zenodo: request error {e}"
        ) from e

    if not response.ok:
        try:
            msg = response.json()
        except Exception:
            msg = response.text
        raise ServerError(
            f"Error while fetching {url} from Zenodo: "
            f"({response.status_code}) {msg}"
        )

    try:
        return response.json()
    except Exception as e:
        logger.exception("Error parsing JSON for Zenodo record %s", identifier)
        raise ParsingError(
            f"Error parsing JSON of Zenodo record {identifier}: {response.content}"
        ) from e


def _convert_record_to_aiod(record: dict) -> dict:
    numeric_id = int(record["id"])
    identifier = str(numeric_id)
    metadata = record.get("metadata", {}) or {}
    links = record.get("links", {}) or {}

    title = metadata.get("title") or f"Zenodo record {identifier}"
    if not isinstance(title, str):
        title = str(title)
    if len(title) > 256:
        text_break = " [...]"
        title = title[: 256 - len(text_break)] + text_break

    description = metadata.get("description") or ""
    if not isinstance(description, str):
        logger.warning(
            "Unexpected description type for record %s: %r", identifier, description
        )
        description = str(description)

    if len(description) > MAX_DESCRIPTION_LENGTH:
        text_break = " [...]"
        description = description[: MAX_DESCRIPTION_LENGTH - len(text_break)] + text_break

    date_published = record.get("created")

    license_value = None
    license_meta = metadata.get("license")
    if isinstance(license_meta, dict):
        license_value = (
            license_meta.get("id")
            or license_meta.get("title")
            or license_meta.get("url")
        )
    elif isinstance(license_meta, str):
        license_value = license_meta

    keywords = metadata.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]

    distributions = []

    html_link = links.get("html") or links.get("latest_html")
    if html_link:
        distributions.append(
            {
                "content_url": html_link,
                "encoding_format": "text/html",
            }
        )

    files = record.get("files") or []
    for f in files:
        f_links = f.get("links") or {}
        file_url = f_links.get("download") or f_links.get("self")
        if not file_url:
            continue
        encoding_format = f.get("type") or "application/octet-stream"
        distributions.append(
            {
                "content_url": file_url,
                "encoding_format": encoding_format,
            }
        )

    platform_identifier = f"zenodo.org:{identifier}"

    return {
        "platform": PLATFORM_NAME,
        "platform_resource_identifier": platform_identifier,
        "name": title,
        "version": metadata.get("version"),
        "same_as": html_link or links.get("self"),
        "description": {"plain": description},
        "date_published": date_published,
        "license": license_value,
        "distribution": distributions,
        "is_accessible_for_free": True,
        "keyword": keywords,
        "size": None,
    }


def upsert_dataset(record: dict) -> int:
    identifier = str(record["id"])

    try:
        local_dataset = _convert_record_to_aiod(record)
        platform_identifier = local_dataset["platform_resource_identifier"]

        try:
            aiod_dataset = aiod.datasets.get_asset_from_platform(
                platform=PLATFORM_NAME,
                platform_identifier=platform_identifier,
                data_format="json",
            )
        except KeyError:
            aiod_dataset = None
        except RequestsJSONDecodeError as e:
            logger.warning(
                "Non-JSON response when checking existing Zenodo asset %s in AIoD: %s. "
                "Treating as not found and attempting registration.",
                platform_identifier,
                e,
            )
            aiod_dataset = None

        if aiod_dataset is None:
            response = aiod.datasets.register(metadata=local_dataset)
            if isinstance(response, str):
                logger.debug("Indexed Zenodo record %s: %s", identifier, response)
                return HTTPStatus.OK
            elif isinstance(response, requests.Response):
                logger.warning(
                    "Error uploading Zenodo record %s "
                    "(%s): %s",
                    identifier,
                    response.status_code,
                    response.content,
                )
                return response.status_code
            raise RuntimeError(
                f"Unexpected response type from aiod.datasets.register for {identifier}: "
                f"{type(response)}"
            )

        if "identifier" not in aiod_dataset:
            raise RuntimeError(
                "Unexpected server response retrieving Zenodo dataset "
                f"{identifier} from AI-on-Demand: {aiod_dataset}"
            )

        response = aiod.datasets.replace(
            identifier=aiod_dataset["identifier"], metadata=local_dataset
        )
        if response.status_code == HTTPStatus.OK:
            logger.debug(
                "Updated Zenodo record %s as AIoD dataset %s",
                identifier,
                aiod_dataset["identifier"],
            )
        else:
            logger.warning(
                "Could not update AIoD dataset %s for Zenodo record %s "
                "(%s): %s",
                aiod_dataset["identifier"],
                identifier,
                response.status_code,
                response.content,
            )
        return response.status_code

    except Exception as e:
        logger.exception(
            "Exception encountered when upserting Zenodo record %s.", identifier, exc_info=e
        )
        if STOP_ON_UNEXPECTED_ERROR:
            raise
        return -1


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["all", "since", "id"],
        required=True,
        help=(
            "'id': index a single Zenodo record by numeric id. "
            "'since': index all records with id >= VALUE (or 'auto'). "
            "'all': index all records."
        ),
    )
    parser.add_argument(
        "--value",
        default=None,
        required=False,
        type=str,
        help=(
            "For mode 'id' this must be a Zenodo numeric identifier. "
            "For mode 'since' this must be a numeric id, or 'auto' to start "
            "from the newest record already indexed in AIoD. "
            "Must not be set for mode 'all'."
        ),
    )

    log_levels = [level.lower() for level in logging.getLevelNamesMapping()]
    parser.add_argument(
        "--app-log-level",
        choices=log_levels,
        default="info",
        help="Emit log messages of at least this level for the connector itself.",
    )
    parser.add_argument(
        "--root-log-level",
        choices=log_levels,
        default="error",
        help="Emit log messages of at least this level for dependencies.",
    )

    args = parser.parse_args()

    if args.mode == "all" and args.value:
        logger.error("Cannot run mode 'all' when a value is supplied.")
        raise SystemExit(1)

    return args


def configure_connector():
    global PLATFORM_NAME, STOP_ON_UNEXPECTED_ERROR, PER_DATASET_DELAY, ZENODO_PAGE_SIZE

    dot_file = Path("~/.aiod/zenodo/.env").expanduser()
    if dot_file.exists() and load_dotenv(dot_file):
        logger.info("Loaded variables from %s", dot_file)
    else:
        logger.info("No environment variables loaded from %s.", dot_file)

    PLATFORM_NAME = os.getenv("PLATFORM_NAME", PLATFORM_NAME)

    delay = os.getenv("PER_DATASET_DELAY")
    PER_DATASET_DELAY = float(delay) if delay else None

    STOP_ON_UNEXPECTED_ERROR = (
        str(os.getenv("STOP_ON_UNEXPECTED_ERROR", str(STOP_ON_UNEXPECTED_ERROR))).lower()
        == "true"
    )

    page_size_env = os.getenv("ZENODO_PAGE_SIZE")
    if page_size_env:
        try:
            ZENODO_PAGE_SIZE = int(page_size_env)
        except ValueError:
            logger.warning(
                "Invalid ZENODO_PAGE_SIZE=%r; falling back to %d",
                page_size_env,
                ZENODO_PAGE_SIZE,
            )

    token = os.getenv("CLIENT_SECRET")
    assert token, "CLIENT_SECRET environment variable not set"

    masked_token = "*" * max(4, (len(token) - 4)) + token[-4:]

    logger.info("%-25s %s", "aiondemand version:", aiod.version)
    logger.info("%-25s %s", "STOP_ON_UNEXPECTED_ERROR:", STOP_ON_UNEXPECTED_ERROR)
    logger.info("%-25s %s", "PER_DATASET_DELAY:", PER_DATASET_DELAY)
    logger.info("%-25s %s", "AI-on-Demand API server:", aiod.config.api_server)
    logger.info("%-25s %s", "Platform Name:", PLATFORM_NAME)
    logger.info("%-25s %s", "Authentication server:", aiod.config.auth_server)
    logger.info("%-25s %s", "Client ID:", aiod.config.client_id)
    logger.info("%-25s %s", "Using secret:", masked_token)

    set_token(Token(client_secret=token))

    logger.info(
        "Configured AI-on-Demand client token for Zenodo connector "
        "(skipping authorization_test)."
    )


def get_newest_indexed_record() -> str:
    logger.info("Finding last uploaded Zenodo record on AI-on-Demand")
    last_id = 0
    batch_size = 100

    for offset in range(0, 1_000_000, batch_size):
        datasets = aiod.datasets.get_list(
            platform=PLATFORM_NAME,
            data_format="json",
            offset=offset,
            limit=batch_size,
        )
        if not datasets:
            break

        for d in datasets:
            pri = d.get("platform_resource_identifier")
            if not isinstance(pri, str):
                continue
            numeric_part = "".join(ch for ch in pri.split(":")[-1] if ch.isdigit())
            if not numeric_part:
                continue
            try:
                last_id = max(last_id, int(numeric_part))
            except ValueError:
                continue

        logger.info("Current highest Zenodo id seen: %s", last_id)

    logger.info("Newest indexed Zenodo id in AIoD: %s", last_id)
    return str(last_id)


def main():
    args = parse_args()

    logging.basicConfig(level=args.root_log_level.upper())
    logger.setLevel(args.app_log_level.upper())

    configure_connector()

    errors: list[Optional[Exception]] = []

    mode = args.mode
    value = args.value

    if mode == "id":
        if not value or not value.isdigit():
            logger.error("For mode 'id', --value must be an integer Zenodo id, got %r", value)
            raise SystemExit(1)

        record = fetch_zenodo_record(int(value))
        upsert_dataset(record)

    elif mode == "since":
        if value == "auto":
            value = get_newest_indexed_record()

        if not value or not value.isdigit():
            logger.error(
                "For mode 'since', --value must be an integer Zenodo id or 'auto', got %r",
                value,
            )
            raise SystemExit(1)

        start_id = int(value)
        logger.info("Indexing Zenodo records with id >= %s", start_id)

        for record in list_records(from_id=start_id):
            try:
                upsert_dataset(record)
                errors.append(None)
            except Exception as e:
                logger.error("Unrecoverable error upserting Zenodo record %s", record.get("id"))
                logger.exception(e)
                errors.append(e)

            if len(errors) > 10:
                errors.pop(0)
                if sum(e is not None for e in errors) > 5:
                    logger.error(
                        "Stopping because too many errors were encountered when "
                        "indexing Zenodo records."
                    )
                    raise SystemExit(1)

            if PER_DATASET_DELAY:
                time.sleep(PER_DATASET_DELAY)

    elif mode == "all":
        logger.info("Indexing ALL Zenodo records (this may take a very long time)")
        for record in list_records():
            upsert_dataset(record)
            if PER_DATASET_DELAY:
                time.sleep(PER_DATASET_DELAY)
    else:
        raise NotImplementedError(f"Unexpected mode: {mode!r}")


if __name__ == "__main__":
    main()