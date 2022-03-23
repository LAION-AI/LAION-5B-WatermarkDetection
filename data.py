from PIL import Image
from functools import partial
import io
import webdataset as wds


def filter_dataset(
    item,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    caption_key="txt",
    image_key="jpg"
):
    if enable_text and caption_key not in item:
        return False
    if enable_image and image_key not in item:
        return False
    if enable_metadata and "json" not in item:
        return False
    return True

def preprocess_dataset(
    item,
    enable_image=True,
    enable_text=True,
    enable_metadata=False,
    image_key="jpg",
    caption_key="txt",
    image_transform=None
):
    output = {}
    if enable_image:
        image_data = item[image_key]
        image = Image.open(io.BytesIO(image_data))
        image_tensor = image_transform(image)
        output["image_filename"] = item["__key__"]
        output["image_tensor"] = image_tensor

    if enable_text:
        text = item[caption_key]
        caption = text.decode("utf-8")
        output["text"] = caption

    if enable_metadata:
        metadata_file = item["json"]
        metadata = metadata_file.decode("utf-8")
        output["metadata"] = metadata
    return output

def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    input_sampler=lambda a: a,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""

    urls = input_sampler(urls)

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)

    filter_dataset = partial(filter_dataset, enable_text=enable_text, enable_image=enable_image, enable_metadata=enable_metadata, caption_key=caption_key, image_key=image_key)
    filtered_dataset = dataset.select(filter_dataset)

    preprocess_dataset = partial(preprocess_dataset, enable_image=enable_image, enable_text=enable_text, enable_metadata=enable_metadata, image_key=image_key, caption_key=caption_key, image_transform=image_transform)
    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset