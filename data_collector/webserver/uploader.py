import requests

def upload_file(label, remote_url, server_url):
    try:
        # Fetch the image from the remote server without saving it locally
        remote_image_response = requests.get(remote_url)

        if remote_image_response.status_code == 200:
            # Upload the image to your server without saving it locally
            upload_response = requests.post(
                server_url,
                files={"file": (label, remote_image_response.content, "image/jpeg")},
            )

            if upload_response.status_code == 200:
                print("Image uploaded successfully to your server!")
            else:
                print(f"Failed to upload image. Status code: {upload_response.status_code}")

        else:
            print(f"Failed to fetch image. Status code: {remote_image_response.status_code}")

    except requests.RequestException as e:
        print(f"Request Exception: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a file to your server")
    parser.add_argument("-f", "--file", help="File to upload", required=True)
    parser.add_argument("-s", "--server", help="Server URL", required=True)
    args = parser.parse_args()

    localserver_url = "http://localhost:5000/upload"
    upload_file(args.file, args.server, localserver_url)


