from __future__ import print_function

import google.auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/ruben/test.json"


# 13uUOPrVxzqs5ZBk0fNPrzn5mTMEJ6Ih3


# def upload_basic(name):
#     """Insert new file.
#     Returns : Id's of the file uploaded
#
#     Load pre-authorized user credentials from the environment.
#     TODO(developer) - See https://developers.google.com/identity
#     for guides on implementing OAuth2 for the application.
#     """
#     creds, _ = google.auth.default()
#
#     try:
#         # create drive api client
#         service = build('drive', 'v3', credentials=creds)
#
#         file_metadata = {'name': name}
#         media = MediaFileUpload(name,
#                                 mimetype='image/png')
#         # pylint: disable=maybe-no-member
#         file = service.files().create(body=file_metadata, media_body=media,
#                                       fields='id').execute()
#         print(F'File ID: {file.get("id")}')
#
#     except HttpError as error:
#         print(F'An error occurred: {error}')
#         file = None
#
#     return file.get('id')
#
#
# if __name__ == '__main__':
#     upload_basic(name='/home/ruben/Pictures/test_image.png')



import io
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def download_file(real_file_id):
    """Downloads a file
    Args:
        real_file_id: ID of the file to download
    Returns : IO object with location.

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    creds, _ = google.auth.default()

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_id = real_file_id

        # pylint: disable=maybe-no-member
        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(F'Download {int(status.progress() * 100)}.')

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    return file.getvalue()


if __name__ == '__main__':
    a=download_file(real_file_id='13uUOPrVxzqs5ZBk0fNPrzn5mTMEJ6Ih3')
    from PIL import Image
    image = Image.open(io.BytesIO(a))
    image.save('test_image_d.png')