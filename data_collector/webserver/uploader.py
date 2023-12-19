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


# Upload the image to your server
your_label = "my_image.jpg"
remote_image_url = "your-image-url" 
your_server_url = "http://localhost:5000/upload"

upload_file(your_label, remote_image_url, your_server_url)
