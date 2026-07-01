import logging
import os

import requests

from odp.config import config

logger = logging.getLogger(__name__)

NEXTCLOUD_FOLDER = "/UPLOAD/ODP Data Submissions/"


def upload_file_to_nextcloud(local_path_to_file, folder_name, file_name):
    if not os.path.exists(local_path_to_file):
        logger.error(f"Error: Local file not found at '{local_path_to_file}'")
        return False

    _clear_and_create_folder(folder_name)

    webdav_url = f"{config.NEXTCLOUD.URL}{NEXTCLOUD_FOLDER}{folder_name}/{file_name}"

    try:
        with open(local_path_to_file, 'rb') as f:
            response = requests.put(
                webdav_url,
                data=f,
                auth=(config.NEXTCLOUD.USER, config.NEXTCLOUD.PASSWORD),
            )

        if response.status_code == 201 or response.status_code == 204:
            logger.info(f"Success! File uploaded. Status code: {response.status_code}")
            return True
        else:
            logger.error(f"Error during upload. Status code: {response.status_code}. Response body: {response.text}")

    except requests.exceptions.RequestException as e:
        logger.exception(f"Error during upload: {e}")

    return False


def delete_folder_from_nextcloud(folder_name):
    """Deletes the folder from nextcloud including its contents"""
    folder_url = f"{config.NEXTCLOUD.URL}{NEXTCLOUD_FOLDER}{folder_name}/"
    auth = (config.NEXTCLOUD.USER, config.NEXTCLOUD.PASSWORD)

    try:
        requests.delete(folder_url, auth=auth)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting folder: {e}")


def _clear_and_create_folder(folder_name):
    """Deletes the folder and then re-creates it."""
    folder_url = f"{config.NEXTCLOUD.URL}{NEXTCLOUD_FOLDER}{folder_name}/"

    delete_folder_from_nextcloud(folder_name)

    requests.request(
        "MKCOL",
        folder_url,
        auth=(config.NEXTCLOUD.USER, config.NEXTCLOUD.PASSWORD)
    )