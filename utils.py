from PIL import Image
import io


def read_imagefile(file) -> Image.Image:
    return Image.open(io.BytesIO(file)).convert("RGB")
