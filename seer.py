#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
import ccxt
import pprint
from lstm import lstm_type_1, lstm_type_2, lstm_type_3
from marionette import marrionette_type_1
from tfann import tfann_type_1
from cnn import cnn_type_1
import httplib2
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import googleapiclient.http
import oauth2client.client
import io


def SigHandler_SIGINT(signum, frame):
    print()
    sys.exit(0)


class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--which", type=str, help="which one to run")
        parser.add_argument(
            "--download", type=str, help="file name to download"
        )
        parser.add_argument("--what", type=str, help="train or predict")
        parser.add_argument(
            "--pysrcupdate",
            type=str,
            nargs="+",
            help="name of source files to update on the drive",
        )
        parser.add_argument(
            "--gpu",
            action="store_true",
            help="use gpu. if false will use cpu",
            default=False,
        )
        parser.add_argument(
            "--test1", action="store_true", help="test switch 1", default=False
        )
        parser.add_argument(
            "--test2", action="store_true", help="test switch 2", default=False
        )
        parser.add_argument(
            "--dbg", action="store_true", help="debug", default=False
        )
        self.args = parser.parse_args()


def get_name_from_path(path):
    path_pos = path.rfind("/")
    if path_pos == -1:
        return path
    else:
        return path[path_pos + 1 :]


def authenticate_drive():
    OAUTH2_SCOPE = "https://www.googleapis.com/auth/drive"
    CLIENT_SECRETS = "./secret.json"
    flow = oauth2client.client.flow_from_clientsecrets(
        CLIENT_SECRETS, OAUTH2_SCOPE
    )
    flow.redirect_uri = oauth2client.client.OOB_CALLBACK_URN
    authorize_url = flow.step1_get_authorize_url()
    print("Go to the following link in your browser: " + authorize_url)
    code = input("Enter verification code: ").strip()
    credentials = flow.step2_exchange(code)
    http = httplib2.Http()
    credentials.authorize(http)
    drive_service = build("drive", "v3", http=http)
    return drive_service


def get_folder_id(folder_name, drive_service):
    parent_dir = (
        drive_service.files()
        .list(
            q="mimeType='application/vnd.google-apps.folder' and name='"
            + folder_name
            + "'",
            fields="files(id, name)",
            spaces="drive",
        )
        .execute()
    )
    folder_id = str()
    for file in parent_dir.get("files", []):
        print(file.get("name") + "---" + file.get("id"))
        folder_id = file.get("id")
    return folder_id


def get_file_id(file_name, folder_name, drive_service):
    folder_id = get_folder_id(folder_name, drive_service)
    download_to_be = (
        drive_service.files()
        .list(
            q="name='" + file_name + "' and '" + folder_id + "' in parents",
            fields="files(id, name)",
            spaces="drive",
        )
        .execute()
    )
    file_id = str()
    for file in download_to_be.get("files", []):
        print(file.get("name") + "---" + file.get("id"))
        file_id = file.get("id")
    return file_id


def g_drive_up(file_path, file_name, file_type, to_folder):
    FILENAME = file_path
    MIMETYPE = file_type
    TITLE = file_name
    DESCRIPTION = "a file"
    drive_service = authenticate_drive()

    media_body = googleapiclient.http.MediaFileUpload(
        FILENAME, mimetype=MIMETYPE, resumable=True
    )
    parent_dir = (
        drive_service.files()
        .list(
            q="mimeType='application/vnd.google-apps.folder' and name='"
            + to_folder
            + "'",
            fields="files(id, name)",
            spaces="drive",
        )
        .execute()
    )
    folder_id = str()
    for file in parent_dir.get("files", []):
        print(file.get("name") + "---" + file.get("id"))
        folder_id = file.get("id")
    body = {"name": TITLE, "description": DESCRIPTION, "parents": [folder_id]}
    new_file = (
        drive_service.files()
        .create(body=body, media_body=media_body, fields="id")
        .execute()
    )
    print(new_file.get("id"))
    # pprint.pprint(new_file)


def g_drive_down(folder_name, file_name):
    drive_service = authenticate_drive()

    # get folder id
    parent_dir = (
        drive_service.files()
        .list(
            q="mimeType='application/vnd.google-apps.folder' and name='"
            + folder_name
            + "'",
            fields="files(id, name)",
            spaces="drive",
        )
        .execute()
    )
    folder_id = str()
    for file in parent_dir.get("files", []):
        print(file.get("name") + "---" + file.get("id"))
        folder_id = file.get("id")

    # get file id
    download_to_be = (
        drive_service.files()
        .list(
            q="name='" + file_name + "' and '" + folder_id + "' in parents",
            fields="files(id, name)",
            spaces="drive",
        )
        .execute()
    )
    file_id = str()
    for file in download_to_be.get("files", []):
        print(file.get("name") + "---" + file.get("id"))
        file_id = file.get("id")

    request = drive_service.files().get_media(fileId=file_id)
    # fh = io.BytesIO()
    fh = io.FileIO(file_name, "w")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    # print(downloader)


def g_drive_update(folder_name, file_name):
    drive_service = authenticate_drive()
    file_id = get_file_id(file_name, folder_name, drive_service)
    u_file = drive_service.files().get(fileId=file_id).execute()
    media_body = MediaFileUpload(resumable=True)
    updated_file = (
        drive_service.files()
        .update(fileId=file_id, body=u_file, media_body=media_body)
        .execute()
    )


def launch_ais(which):
    if which == "marionette":
        marrionette_type_1()
    elif which == "lstm_type_1":
        lstm_type_1("ethereum", "ether")
    elif which == "lstm_type_2":
        lstm_type_2("ethereum", "ether", 5, 20)
    elif which == "lstm_type_3":
        lstm_type_3("ethereum", "ether", 5, 20)
    elif which == "cnn_type_1":
        cnn_type_1()
    elif which == "tfann_type_1":
        tfann_type_1()
    else:
        pass


# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    # here
    if argparser.args.pysrcupdate:
        for src in argparser.args.pysrcupdate:
            g_drive_up(src, get_name_from_path(src), "text/python", "colab")
    if argparser.args.download:
        g_drive_down("colab", "main.py")
    launch_ais(argparser.args.which)


def main():
    argparser = Argparser()
    if argparser.args.dbg:
        try:
            premain(argparser)
        except Exception as e:
            print(e.__doc__)
            if e.message:
                print(e.message)
            variables = globals().copy()
            variables.update(locals())
            shell = code.InteractiveConsole(variables)
            shell.interact(banner="DEBUG REPL")
    else:
        premain(argparser)


if __name__ == "__main__":
    main()
