import requests

from PIL import Image
import io

def convert_rgb565_to_png(data): # assumin96x96 image
    # Convert the image to RGB888
    rgb888 = bytearray(96 * 96 * 3)
    for i in range(96 * 96):
        r5 = (data[i * 2] & 0xF8) >> 3 
        g6 = ((data[i * 2] & 0x07) << 3) | ((data[i * 2 + 1] & 0xE0) >> 5)
        b5 = (data[i * 2 + 1] & 0x1F)
        
        rgb888[i * 3] = (r5 * 255) // 31
        rgb888[i * 3 + 1] = (g6 * 255) // 63
        rgb888[i * 3 + 2] = (b5 * 255) // 31

    image_png = Image.frombytes("RGB", (96, 96), bytes(rgb888))
    with io.BytesIO() as f:
        image_png.save(f, format="PNG")
        return f.getvalue()

def upload_file(label, remote_url, server_url):
    try:
        # Fetch the image from the remote server without saving it locally
        remote_image_response = requests.get(remote_url)

        # Convert the image to PNG
        img_data = convert_rgb565_to_png(remote_image_response.content)

        if remote_image_response.status_code == 200:
            # Upload the image to your server without saving it locally
            upload_response = requests.post(
                server_url,
                files={"file": (label, img_data, "image/png")},
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


